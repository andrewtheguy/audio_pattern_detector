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
/// number of templates: the signal is transformed once by [`load_signal`]
/// and each [`correlate`] call then costs a spectrum product and one
/// inverse FFT.
///
/// [`load_signal`]: CorrelationWorkspace::load_signal
/// [`correlate`]: CorrelationWorkspace::correlate
#[derive(Default)]
pub struct CorrelationWorkspace {
    fft_size: usize,
    signal_len: usize,
    max_template_len: usize,
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

    /// Transform `signal`, sizing the FFT so that templates up to
    /// `max_template_len` samples correlate without wrap-around.
    pub fn load_signal(&mut self, signal: &[f32], max_template_len: usize) {
        self.signal_len = signal.len();
        self.max_template_len = max_template_len;
        if signal.is_empty() {
            return;
        }

        // Covers the full correlation (signal + template - 1) and, for an
        // empty template, still the whole signal.
        let fft_size = (signal.len() + max_template_len.saturating_sub(1)).next_power_of_two();
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
        self.padded_signal[..signal.len()].copy_from_slice(signal);

        let (r2c, _) = self.plans.as_ref().expect("plans were just set");
        r2c.process_with_scratch(&mut self.padded_signal, &mut self.signal_spectrum, &mut self.forward_scratch)
            .expect("forward FFT buffers are sized by the plan");
    }

    /// Full cross-correlation of the loaded signal with `template`, written
    /// to `output` (length `signal.len() + template.len() - 1`).
    ///
    /// Matches `scipy.signal.correlate(signal, template, mode="full")`:
    /// output index `k` is the lag where `template[template.len() - 1]`
    /// aligns with `signal[k]`. `output` is empty if either is empty.
    pub fn correlate(&mut self, template: &mut CorrelationTemplate, output: &mut Vec<f32>) {
        output.clear();
        if self.signal_len == 0 || template.is_empty() {
            return;
        }
        assert!(
            template.len() <= self.max_template_len,
            "template longer than the loaded signal allows"
        );

        let template_spectrum = template.spectrum(self.fft_size);
        for ((p, s), t) in self.product.iter_mut().zip(&self.signal_spectrum).zip(template_spectrum) {
            *p = *s * *t;
        }

        let (_, c2r) = self.plans.as_ref().expect("a signal is loaded");
        c2r.process_with_scratch(&mut self.product, &mut self.time_domain, &mut self.inverse_scratch)
            .expect("inverse FFT buffers are sized by the plan");

        let output_len = self.signal_len + template.len() - 1;
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
    workspace.load_signal(signal, template.len());
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
            workspace.load_signal(signal, template_b.len());
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
        workspace.load_signal(&[], 2);
        workspace.correlate(&mut template, &mut output);
        assert!(output.is_empty());

        let mut empty = CorrelationTemplate::new(&[]);
        workspace.load_signal(&[1.0, 2.0, 3.0], 2);
        workspace.correlate(&mut empty, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_autocorrelation_peak_is_centered() {
        let clip = [0.5_f32, -1.0, 0.25, 0.75];
        let corr = fft_correlate_full(&clip, &clip);
        let energy: f32 = clip.iter().map(|v| v * v).sum();
        assert!((corr[clip.len() - 1] - energy).abs() < 1e-5);
    }
}
