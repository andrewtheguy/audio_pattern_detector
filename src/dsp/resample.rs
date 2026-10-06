use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// ── Resampling ───────────────────────────────────────────────────────

/// FFT-based resampling of a 1-D signal, matching `scipy.signal.resample`.
///
/// Uses the full complex FFT to truncate or zero-pad the frequency-domain
/// representation, exactly replicating scipy's spectrum manipulation.
pub fn resample_1d(data: &[f32], target_len: usize) -> Vec<f32> {
    let n = data.len();
    if n == 0 || target_len == 0 {
        return vec![0.0; target_len];
    }
    if n == target_len {
        return data.to_vec();
    }

    let m = target_len;

    // Forward complex FFT.
    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(n);
    let mut spectrum: Vec<Complex<f64>> =
        data.iter().map(|&v| Complex::new(v as f64, 0.0)).collect();
    fft_fwd.process(&mut spectrum);

    // Build new spectrum of length m, matching scipy's slice logic:
    //   N = min(num, Nx)
    //   Y[0:N//2+1]      = X[0:N//2+1]       (positive frequencies and Nyquist)
    //   Y[-(N-1)//2:]    = X[-(N-1)//2:]      (negative frequencies)
    let n_common = n.min(m);
    let pos = n_common / 2 + 1; // positive-frequency bins to copy, including Nyquist if present
    let neg = (n_common - 1) / 2; // number of negative-frequency bins to copy

    let mut new_spectrum = vec![Complex::new(0.0, 0.0); m];
    new_spectrum[..pos].copy_from_slice(&spectrum[..pos]);
    if neg > 0 {
        new_spectrum[m - neg..].copy_from_slice(&spectrum[n - neg..]);
    }
    if n_common.is_multiple_of(2) {
        let nyquist = n_common / 2;
        if m < n {
            // Downsampling: the output Nyquist bin holds both input components.
            new_spectrum[nyquist] += spectrum[n - nyquist];
        } else {
            // Upsampling: split the input Nyquist component across both bins.
            new_spectrum[nyquist] *= 0.5;
            new_spectrum[m - nyquist] = new_spectrum[nyquist];
        }
    }

    // Inverse complex FFT.
    let fft_inv = planner.plan_fft_inverse(m);
    fft_inv.process(&mut new_spectrum);

    // Scale: rustfft inverse is un-normalised (factor m), and scipy applies
    // target/source.  Combined scale: target / (source * target) = 1/source.
    let scale = 1.0 / n as f64;
    new_spectrum.iter().map(|c| (c.re * scale) as f32).collect()
}

/// Resample a 1-D signal to `target_len` by partitioning it into windows
/// and keeping the maximum sample from each window.
///
/// Works for both downsampling and upsampling.  Guarantees
/// `output.len() == target_len`.  When upsampling, windows that map to the
/// same source sample simply repeat it.
pub fn resample_preserve_maxima_1d(data: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 || data.is_empty() {
        return Vec::new();
    }

    let n_points = data.len();
    let step_size = n_points as f64 / target_len as f64;
    let mut downsampled = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let mut start_index = (i as f64 * step_size) as usize;
        let mut end_index = ((i + 1) as f64 * step_size) as usize;

        // Guarantee at least one sample per window
        if end_index <= start_index {
            end_index = start_index + 1;
        }

        // Clamp into [0, n_points)
        if start_index >= n_points {
            start_index = n_points - 1;
        }
        if end_index > n_points {
            end_index = n_points;
        }

        let max_value = data[start_index..end_index]
            .iter()
            .copied()
            .reduce(f32::max)
            .expect("non-empty window must have a maximum");
        downsampled.push(max_value);
    }

    downsampled
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── resample ─────────────────────────────────────────────────────

    #[test]
    fn test_resample_identity() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let out = resample_1d(&data, 4);
        for (a, b) in out.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[test]
    fn test_resample_empty() {
        assert_eq!(resample_1d(&[], 0), Vec::<f32>::new());
        assert_eq!(resample_1d(&[], 5), vec![0.0; 5]);
        assert_eq!(resample_1d(&[1.0, 2.0], 0), Vec::<f32>::new());
    }

    #[test]
    fn test_resample_downsample() {
        // Sine wave at 8 points → 4 points.
        let n = 8;
        let data: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n as f32).sin())
            .collect();
        let out = resample_1d(&data, 4);
        assert_eq!(out.len(), 4);
        // A single-cycle sine resampled to 4 points should still be ~sinusoidal.
        // Values should be close to sin(0), sin(pi/2), sin(pi), sin(3pi/2) = 0, 1, 0, -1
        assert!(out[0].abs() < 0.1);
        assert!((out[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_resample_upsample() {
        let data = [0.0_f32, 1.0, 0.0];
        let out = resample_1d(&data, 6);
        assert_eq!(out.len(), 6);
    }

    fn assert_resample(data: &[f32], target_len: usize, expected: &[f32]) {
        let out = resample_1d(data, target_len);
        assert_eq!(out.len(), expected.len());
        for (i, (a, b)) in out.iter().zip(expected).enumerate() {
            assert!((a - b).abs() < 1e-5, "index {i}: {out:?} != {expected:?}");
        }
    }

    // Expected values below come from scipy.signal.resample (scipy 1.18.1).

    #[test]
    fn test_resample_even_downsample_combines_nyquist() {
        assert_resample(&[1.0, 0.0, -1.0, 0.0], 2, &[1.0, -1.0]);
        assert_resample(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            6,
            &[1.5, 2.5432043, 3.2752552, 5.5, 5.724745, 8.456796],
        );
    }

    #[test]
    fn test_resample_even_upsample_splits_nyquist() {
        assert_resample(&[1.0, -1.0], 4, &[1.0, 0.0, -1.0, 0.0]);
        assert_resample(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            8,
            &[1.0, 1.3443514, 2.767949, 3.2061589, 4.0, 4.500948, 6.232051, 4.9485416],
        );
    }

    #[test]
    fn test_resample_odd_lengths_match_scipy() {
        assert_resample(
            &[3.0, 1.0, 4.0, 1.0, 5.0],
            8,
            &[3.0, 0.36985505, 2.1072586, 4.0756545, 2.3527863, 1.0666965, 3.739955, 5.687794],
        );
        assert_resample(
            &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0],
            4,
            &[0.98076135, 3.0706325, 2.9396849, 7.294636],
        );
        assert_resample(
            &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
            3,
            &[2.8446698, 2.8329673, 5.947363],
        );
    }

    #[test]
    fn test_resample_preserve_maxima_identity() {
        let data = [1.0_f32, 3.0, 2.0, 4.0];
        let out = resample_preserve_maxima_1d(&data, data.len());
        assert_eq!(out, data);
    }

    #[test]
    fn test_resample_preserve_maxima_window_maxes() {
        let data = [1.0_f32, 5.0, 2.0, 4.0, 3.0, 6.0];
        let out = resample_preserve_maxima_1d(&data, 3);
        assert_eq!(out, vec![5.0, 4.0, 6.0]);
    }

    #[test]
    fn test_resample_preserve_maxima_short_input() {
        // Upsampling: 3 samples → 5 windows (step_size = 0.6).
        // i=0: [0..1)→1.0, i=1: [0..1)→1.0, i=2: [1..2)→2.0,
        // i=3: [1..2)→2.0, i=4: [2..3)→3.0
        let data = [1.0_f32, 2.0, 3.0];
        let out = resample_preserve_maxima_1d(&data, 5);
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resample_preserve_maxima_upsample_single() {
        // Edge case: 1 sample → 4 windows should repeat the value.
        let data = [7.0_f32];
        let out = resample_preserve_maxima_1d(&data, 4);
        assert_eq!(out, vec![7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_resample_preserve_maxima_upsample_two_to_six() {
        // 2 samples → 6 windows (step_size = 0.333): each source sample
        // maps to 3 windows.
        let data = [1.0_f32, 5.0];
        let out = resample_preserve_maxima_1d(&data, 6);
        assert_eq!(out, vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_resample_preserve_maxima_upsample_preserves_all_values() {
        // 5 samples → 20 windows (step_size = 0.25): each source sample
        // maps to exactly 4 windows.
        let data = [3.0_f32, 1.0, 4.0, 1.0, 5.0];
        let out = resample_preserve_maxima_1d(&data, 20);
        assert_eq!(
            out,
            vec![
                3.0, 3.0, 3.0, 3.0,
                1.0, 1.0, 1.0, 1.0,
                4.0, 4.0, 4.0, 4.0,
                1.0, 1.0, 1.0, 1.0,
                5.0, 5.0, 5.0, 5.0,
            ]
        );
    }

    #[test]
    fn test_resample_preserve_maxima_same_length() {
        // target_len == data.len() should be identity.
        let data = [2.0_f32, 8.0, 3.0, 7.0, 1.0];
        let out = resample_preserve_maxima_1d(&data, 5);
        assert_eq!(out, data);
    }

    #[test]
    fn test_resample_preserve_maxima_empty_and_zero_target() {
        assert_eq!(resample_preserve_maxima_1d(&[], 4), Vec::<f32>::new());
        assert_eq!(
            resample_preserve_maxima_1d(&[1.0, 2.0], 0),
            Vec::<f32>::new()
        );
    }
}
