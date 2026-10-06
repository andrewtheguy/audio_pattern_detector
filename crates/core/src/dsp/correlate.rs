// ── FFT cross-correlation ────────────────────────────────────────────

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

/// Cross-correlate two real 1-D signals using FFT, returning the full
/// correlation (length `signal.len() + template.len() - 1`).
///
/// Matches `scipy.signal.correlate(signal, template, mode="full")`: output
/// index `k` is the lag where `template[template.len() - 1]` aligns with
/// `signal[k]`.  Returns an empty vector if either input is empty.
pub fn fft_correlate_full(signal: &[f32], template: &[f32]) -> Vec<f32> {
    if signal.is_empty() || template.is_empty() {
        return Vec::new();
    }

    let output_len = signal.len() + template.len() - 1;
    let fft_size = output_len.next_power_of_two();

    let mut padded_signal = vec![0.0_f32; fft_size];
    let mut padded_template = vec![0.0_f32; fft_size];
    padded_signal[..signal.len()].copy_from_slice(signal);
    // Time-reversing the template turns convolution into correlation.
    for (dst, &val) in padded_template.iter_mut().zip(template.iter().rev()) {
        *dst = val;
    }

    let (r2c, c2r) = get_fft_plans(fft_size);
    let mut signal_spectrum = r2c.make_output_vec();
    let mut template_spectrum = r2c.make_output_vec();
    let mut forward_scratch = r2c.make_scratch_vec();

    r2c.process_with_scratch(&mut padded_signal, &mut signal_spectrum, &mut forward_scratch)
        .expect("forward FFT buffers are sized by the plan");
    r2c.process_with_scratch(&mut padded_template, &mut template_spectrum, &mut forward_scratch)
        .expect("forward FFT buffers are sized by the plan");

    for (s, t) in signal_spectrum.iter_mut().zip(template_spectrum.iter()) {
        *s *= *t;
    }

    let mut result = c2r.make_output_vec();
    let mut inverse_scratch = c2r.make_scratch_vec();
    c2r.process_with_scratch(&mut signal_spectrum, &mut result, &mut inverse_scratch)
        .expect("inverse FFT buffers are sized by the plan");

    result.truncate(output_len);
    let normalization = 1.0 / fft_size as f32;
    result.iter_mut().for_each(|x| *x *= normalization);
    result
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
    fn test_autocorrelation_peak_is_centered() {
        let clip = [0.5_f32, -1.0, 0.25, 0.75];
        let corr = fft_correlate_full(&clip, &clip);
        let energy: f32 = clip.iter().map(|v| v * v).sum();
        assert!((corr[clip.len() - 1] - energy).abs() < 1e-5);
    }
}
