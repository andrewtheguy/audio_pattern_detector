// ── FFT cross-correlation ────────────────────────────────────────────

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;

const FFT_PLAN_CACHE_CAPACITY: usize = 8;

type FftPlans = (Arc<dyn RealToComplex<f32>>, Arc<dyn ComplexToReal<f32>>);

// Bounded per-thread cache of FFT plans keyed by size, most recently used first.
thread_local! {
    static FFT_PLAN_CACHE: RefCell<VecDeque<(usize, FftPlans)>> =
        RefCell::new(VecDeque::with_capacity(FFT_PLAN_CACHE_CAPACITY));
}

fn get_fft_plans(fft_size: usize) -> FftPlans {
    FFT_PLAN_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        if let Some(index) = cache.iter().position(|(size, _)| *size == fft_size) {
            let entry = cache.remove(index).expect("cache entry exists at located index");
            let plans = entry.1.clone();
            cache.push_front(entry);
            return plans;
        }

        // Use a fresh planner per cache miss so planner-internal maps do not grow unbounded.
        let mut planner = RealFftPlanner::<f32>::new();
        let plans: FftPlans = (
            planner.plan_fft_forward(fft_size),
            planner.plan_fft_inverse(fft_size),
        );
        if cache.len() == FFT_PLAN_CACHE_CAPACITY {
            cache.pop_back();
        }
        cache.push_front((fft_size, plans.clone()));
        plans
    })
}

/// Number of FFT sizes a [`CorrelationTemplate`] keeps spectra for. A run
/// sees at most three: the first chunk (no lookback), full chunks and the
/// final short chunk.
const TEMPLATE_SPECTRUM_CACHE_CAPACITY: usize = 4;

/// A correlation template with its time-reversed spectrum cached per FFT
/// size, so repeated correlations against new signals only transform the
/// signal.
pub struct CorrelationTemplate {
    reversed: Vec<f32>,
    spectra: Vec<(usize, Vec<Complex<f32>>)>,
}

impl CorrelationTemplate {
    pub fn new(template: &[f32]) -> Self {
        Self {
            // Time-reversing the template turns convolution into correlation.
            reversed: template.iter().rev().copied().collect(),
            spectra: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.reversed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reversed.is_empty()
    }

    /// Spectrum of the zero-padded, reversed template at `fft_size`.
    fn spectrum(&mut self, fft_size: usize) -> &[Complex<f32>] {
        if let Some(index) = self.spectra.iter().position(|(size, _)| *size == fft_size) {
            return &self.spectra[index].1;
        }
        assert!(fft_size >= self.reversed.len(), "FFT size must cover the template");

        let (r2c, _) = get_fft_plans(fft_size);
        let mut padded = vec![0.0_f32; fft_size];
        padded[..self.reversed.len()].copy_from_slice(&self.reversed);
        let mut spectrum = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();
        r2c.process_with_scratch(&mut padded, &mut spectrum, &mut scratch)
            .expect("forward FFT buffers are sized by the plan");

        if self.spectra.len() == TEMPLATE_SPECTRUM_CACHE_CAPACITY {
            self.spectra.remove(0);
        }
        self.spectra.push((fft_size, spectrum));
        &self.spectra.last().expect("spectrum was just pushed").1
    }
}

/// Reusable buffers for FFT cross-correlation of one signal against any
/// number of templates: the signal is set by [`load_signal`] and transformed
/// once per FFT size, so each [`correlate`] call with a template needing the
/// same size costs a spectrum product and one inverse FFT.
///
/// The FFT size of a correlation depends only on the signal and that
/// template, never on the other templates, so the result is bit-identical
/// to correlating the pair on its own.
///
/// [`load_signal`]: CorrelationWorkspace::load_signal
/// [`correlate`]: CorrelationWorkspace::correlate
#[derive(Default)]
pub struct CorrelationWorkspace {
    signal: Vec<f32>,
    /// FFT size the buffers are allocated for (0 before the first transform).
    fft_size: usize,
    /// Whether `signal_spectrum` is the transform of `signal` at `fft_size`.
    spectrum_loaded: bool,
    plans: Option<FftPlans>,
    padded_signal: Vec<f32>,
    signal_spectrum: Vec<Complex<f32>>,
    product: Vec<Complex<f32>>,
    forward_scratch: Vec<Complex<f32>>,
    inverse_scratch: Vec<Complex<f32>>,
    time_domain: Vec<f32>,
}

impl CorrelationWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the signal for the following [`correlate`](Self::correlate) calls.
    pub fn load_signal(&mut self, signal: &[f32]) {
        self.signal.clear();
        self.signal.extend_from_slice(signal);
        self.spectrum_loaded = false;
    }

    /// Transform the loaded signal at `fft_size` unless that is already done.
    fn transform_signal(&mut self, fft_size: usize) {
        if self.spectrum_loaded && self.fft_size == fft_size {
            return;
        }
        if self.fft_size != fft_size || self.plans.is_none() {
            let plans = get_fft_plans(fft_size);
            let (r2c, c2r) = &plans;
            self.signal_spectrum = r2c.make_output_vec();
            self.product = r2c.make_output_vec();
            self.forward_scratch = r2c.make_scratch_vec();
            self.inverse_scratch = c2r.make_scratch_vec();
            self.time_domain = c2r.make_output_vec();
            self.plans = Some(plans);
            self.fft_size = fft_size;
        }

        self.padded_signal.clear();
        self.padded_signal.resize(fft_size, 0.0);
        self.padded_signal[..self.signal.len()].copy_from_slice(&self.signal);

        let (r2c, _) = self.plans.as_ref().expect("plans were just set");
        r2c.process_with_scratch(&mut self.padded_signal, &mut self.signal_spectrum, &mut self.forward_scratch)
            .expect("forward FFT buffers are sized by the plan");
        self.spectrum_loaded = true;
    }

    /// Full cross-correlation of the loaded signal with `template`, written
    /// to `output` (length `signal.len() + template.len() - 1`).
    ///
    /// Matches `scipy.signal.correlate(signal, template, mode="full")`:
    /// output index `k` is the lag where `template[template.len() - 1]`
    /// aligns with `signal[k]`. `output` is empty if either is empty.
    pub fn correlate(&mut self, template: &mut CorrelationTemplate, output: &mut Vec<f32>) {
        output.clear();
        if self.signal.is_empty() || template.is_empty() {
            return;
        }

        // Covers the full correlation without wrap-around.
        let output_len = self.signal.len() + template.len() - 1;
        self.transform_signal(output_len.next_power_of_two());

        let template_spectrum = template.spectrum(self.fft_size);
        for ((p, s), t) in self.product.iter_mut().zip(&self.signal_spectrum).zip(template_spectrum) {
            *p = *s * *t;
        }

        let (_, c2r) = self.plans.as_ref().expect("the signal was just transformed");
        c2r.process_with_scratch(&mut self.product, &mut self.time_domain, &mut self.inverse_scratch)
            .expect("inverse FFT buffers are sized by the plan");

        let normalization = 1.0 / self.fft_size as f32;
        output.extend(self.time_domain[..output_len].iter().map(|x| x * normalization));
    }
}

/// Cross-correlate two real 1-D signals using FFT, returning the full
/// correlation (length `signal.len() + template.len() - 1`).
///
/// One-shot convenience over [`CorrelationWorkspace`]; see
/// [`CorrelationWorkspace::correlate`] for the semantics.
pub fn fft_correlate_full(signal: &[f32], template: &[f32]) -> Vec<f32> {
    let mut workspace = CorrelationWorkspace::new();
    let mut template = CorrelationTemplate::new(template);
    let mut output = Vec::new();
    workspace.load_signal(signal);
    workspace.correlate(&mut template, &mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // Naive correlation used for correctness checks.
    fn naive_full_correlation(signal: &[f32], template: &[f32]) -> Vec<f32> {
        let output_len = signal.len() + template.len() - 1;
        let mut result = vec![0.0; output_len];
        for (lag, out) in result.iter_mut().enumerate() {
            for (i, &t) in template.iter().enumerate() {
                let signal_idx = lag as isize - (template.len() as isize - 1) + i as isize;
                if (0..signal.len() as isize).contains(&signal_idx) {
                    *out += signal[signal_idx as usize] * t;
                }
            }
        }
        result
    }

    #[test]
    fn test_full_length() {
        let result = fft_correlate_full(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 0.0, 0.0]);
        assert_eq!(result.len(), 7);
        assert_eq!(fft_correlate_full(&[1.0; 100], &[1.0; 10]).len(), 109);
    }

    #[test]
    fn test_empty_inputs() {
        assert_eq!(fft_correlate_full(&[], &[1.0]), Vec::<f32>::new());
        assert_eq!(fft_correlate_full(&[1.0], &[]), Vec::<f32>::new());
        assert_eq!(fft_correlate_full(&[1.0, 2.0], &[]), Vec::<f32>::new());
        assert_eq!(fft_correlate_full(&[1.0, 2.0, 3.0], &[]), Vec::<f32>::new());
    }

    #[test]
    fn test_matches_naive() {
        let signal: Vec<f32> = (0..37).map(|i| ((i * 7 % 11) as f32 - 5.0) / 5.0).collect();
        let template: Vec<f32> = (0..9).map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0).collect();
        let fft = fft_correlate_full(&signal, &template);
        let naive = naive_full_correlation(&signal, &template);
        assert_eq!(fft.len(), naive.len());
        for (a, b) in fft.iter().zip(naive.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    fn test_workspace_reuse_matches_one_shot() {
        let signal_a: Vec<f32> = (0..50).map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0).collect();
        let signal_b: Vec<f32> = (0..70).map(|i| ((i * 3 % 19) as f32 - 9.0) / 9.0).collect();
        let template_a: Vec<f32> = (0..7).map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0).collect();
        let template_b: Vec<f32> = (0..12).map(|i| ((i * 11 % 13) as f32 - 6.0) / 6.0).collect();

        let mut workspace = CorrelationWorkspace::new();
        let mut cached_a = CorrelationTemplate::new(&template_a);
        let mut cached_b = CorrelationTemplate::new(&template_b);
        let mut output = Vec::new();

        // Two templates against one signal, then a different signal size.
        for signal in [&signal_a, &signal_b, &signal_a] {
            workspace.load_signal(signal);
            for (cached, template) in [(&mut cached_a, &template_a), (&mut cached_b, &template_b)] {
                workspace.correlate(cached, &mut output);
                let naive = naive_full_correlation(signal, template);
                assert_eq!(output.len(), naive.len());
                for (a, b) in output.iter().zip(naive.iter()) {
                    assert!((a - b).abs() < 1e-4, "{a} != {b}");
                }
            }
        }
        assert_eq!(cached_a.spectra.len(), 2, "one cached spectrum per FFT size");
    }

    #[test]
    fn test_workspace_empty_inputs() {
        let mut workspace = CorrelationWorkspace::new();
        let mut template = CorrelationTemplate::new(&[1.0, 2.0]);
        let mut output = vec![1.0];
        workspace.load_signal(&[]);
        workspace.correlate(&mut template, &mut output);
        assert!(output.is_empty());

        let mut empty = CorrelationTemplate::new(&[]);
        workspace.load_signal(&[1.0, 2.0, 3.0]);
        workspace.correlate(&mut empty, &mut output);
        assert!(output.is_empty());
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    fn test_template_len_and_is_empty() {
        let template = CorrelationTemplate::new(&[1.0, 2.0, 3.0]);
        assert_eq!(template.len(), 3);
        assert!(!template.is_empty());
        assert_eq!(template.reversed, vec![3.0, 2.0, 1.0]);

        let empty = CorrelationTemplate::new(&[]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_cached_spectrum_gives_identical_output() {
        let signal: Vec<f32> = (0..90).map(|i| ((i * 29 % 31) as f32 - 15.0) / 15.0).collect();
        let template: Vec<f32> = (0..11).map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0).collect();

        let mut workspace = CorrelationWorkspace::new();
        let mut cached = CorrelationTemplate::new(&template);
        let mut first = Vec::new();
        let mut second = Vec::new();

        workspace.load_signal(&signal);
        workspace.correlate(&mut cached, &mut first);
        assert_eq!(cached.spectra.len(), 1);
        // Same signal again: the cached spectrum is used (no new entry) and
        // the result is bit-identical to the first pass and to the one-shot.
        workspace.load_signal(&signal);
        workspace.correlate(&mut cached, &mut second);
        assert_eq!(cached.spectra.len(), 1);
        assert_eq!(first, second);
        assert_eq!(first, fft_correlate_full(&signal, &template));
    }

    #[test]
    fn test_template_spectrum_cache_evicts_oldest() {
        let template: Vec<f32> = (0..5).map(|i| ((i * 3 % 5) as f32 - 2.0) / 2.0).collect();
        let mut cached = CorrelationTemplate::new(&template);
        let mut workspace = CorrelationWorkspace::new();
        let mut output = Vec::new();

        // Six signal lengths, each needing a different power-of-two FFT.
        let lengths = [4, 12, 28, 60, 124, 252];
        let mut fft_sizes = Vec::new();
        for &len in &lengths {
            let signal: Vec<f32> = (0..len).map(|i| ((i * 17 % 23) as f32 - 11.0) / 11.0).collect();
            workspace.load_signal(&signal);
            workspace.correlate(&mut cached, &mut output);
            assert_close(&output, &naive_full_correlation(&signal, &template));
            fft_sizes.push(workspace.fft_size);
            assert!(cached.spectra.len() <= TEMPLATE_SPECTRUM_CACHE_CAPACITY);
        }
        assert_eq!(fft_sizes, vec![8, 16, 32, 64, 128, 256]);

        // The oldest two sizes were evicted, the newest four kept in order.
        let kept: Vec<usize> = cached.spectra.iter().map(|(size, _)| *size).collect();
        assert_eq!(kept, vec![32, 64, 128, 256]);

        // An evicted size is recomputed and still correct.
        let signal: Vec<f32> = (0..4).map(|i| i as f32 - 1.5).collect();
        workspace.load_signal(&signal);
        workspace.correlate(&mut cached, &mut output);
        assert_close(&output, &naive_full_correlation(&signal, &template));
        let kept: Vec<usize> = cached.spectra.iter().map(|(size, _)| *size).collect();
        assert_eq!(kept, vec![64, 128, 256, 8]);
    }

    #[test]
    fn test_fft_size_is_independent_of_other_templates() {
        let signal: Vec<f32> = (0..40).map(|i| ((i * 19 % 29) as f32 - 14.0) / 14.0).collect();
        let short: Vec<f32> = (0..6).map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0).collect();
        let long: Vec<f32> = (0..30).map(|i| ((i * 11 % 17) as f32 - 8.0) / 8.0).collect();
        let expected_short = fft_correlate_full(&signal, &short);
        let expected_long = fft_correlate_full(&signal, &long);

        let mut workspace = CorrelationWorkspace::new();
        let mut output = Vec::new();
        workspace.load_signal(&signal);
        // Each template gets the FFT size it needs on its own, in any order,
        // so the output is bit-identical to the one-shot correlation.
        let mut fft_sizes = Vec::new();
        for (template, expected) in [(&long, &expected_long), (&short, &expected_short), (&long, &expected_long)] {
            workspace.correlate(&mut CorrelationTemplate::new(template), &mut output);
            fft_sizes.push(workspace.fft_size);
            assert_eq!(&output, expected);
            assert_close(&output, &naive_full_correlation(&signal, template));
        }
        assert_eq!(fft_sizes, vec![128, 64, 128]);
    }

    #[test]
    fn test_template_longer_than_signal_matches_naive() {
        let signal = [1.0_f32, -2.0, 0.5];
        let template: Vec<f32> = (0..9).map(|i| ((i * 4 % 9) as f32 - 4.0) / 4.0).collect();
        let result = fft_correlate_full(&signal, &template);
        assert_eq!(result.len(), 11);
        assert_close(&result, &naive_full_correlation(&signal, &template));
    }

    #[test]
    fn test_exact_values_match_scipy() {
        // scipy.signal.correlate([1, 2, 3, 4], [1, 0.5], mode="full")
        // = [0.5, 2.0, 3.5, 5.0, 4.0]
        let result = fft_correlate_full(&[1.0, 2.0, 3.0, 4.0], &[1.0, 0.5]);
        assert_close(&result, &[0.5, 2.0, 3.5, 5.0, 4.0]);
        // Single-sample template scales the signal.
        let result = fft_correlate_full(&[1.0, -2.0, 3.0], &[2.0]);
        assert_close(&result, &[2.0, -4.0, 6.0]);
    }

    #[test]
    fn test_output_buffer_is_replaced_not_appended() {
        let mut workspace = CorrelationWorkspace::new();
        let mut template = CorrelationTemplate::new(&[1.0, 0.5]);
        let mut output = vec![9.0; 20];
        workspace.load_signal(&[1.0, 2.0, 3.0, 4.0]);
        workspace.correlate(&mut template, &mut output);
        assert_close(&output, &[0.5, 2.0, 3.5, 5.0, 4.0]);
    }

    #[test]
    fn test_autocorrelation_peak_is_centered() {
        let clip = [0.5_f32, -1.0, 0.25, 0.75];
        let corr = fft_correlate_full(&clip, &clip);
        let energy: f32 = clip.iter().map(|v| v * v).sum();
        assert!((corr[clip.len() - 1] - energy).abs() < 1e-5);
    }
}
