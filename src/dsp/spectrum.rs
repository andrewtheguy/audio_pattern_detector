// ── Real spectra ─────────────────────────────────────────────────────

use realfft::RealFftPlanner;

/// Hann window of length `n`, matching `numpy.hanning`.
pub fn hanning(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let k = 2.0 * i as f64 - denom;
            0.5 + 0.5 * (std::f64::consts::PI * k / denom).cos()
        })
        .collect()
}

/// Magnitude of the one-sided FFT of a real signal, matching
/// `numpy.abs(numpy.fft.rfft(data))`.  Output length is `data.len() / 2 + 1`.
pub fn rfft_magnitude(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(data.len());
    let mut input = data.to_vec();
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut input, &mut spectrum)
        .expect("FFT buffers are sized by the plan");
    spectrum.iter().map(|c| c.norm()).collect()
}

/// Bin centre frequencies for [`rfft_magnitude`], matching
/// `numpy.fft.rfftfreq(n, d=1 / sample_rate)`.
pub fn rfft_frequencies(n: usize, sample_rate: u32) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let val = 1.0 / (n as f64 * (1.0 / sample_rate as f64));
    (0..n / 2 + 1).map(|k| k as f64 * val).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hanning_matches_numpy() {
        assert_eq!(hanning(0), Vec::<f64>::new());
        assert_eq!(hanning(1), vec![1.0]);
        let w = hanning(5);
        let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
        for (a, b) in w.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn test_rfft_magnitude_of_sine_peaks_at_its_bin() {
        let n = 64;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / n as f64).sin())
            .collect();
        let mag = rfft_magnitude(&data);
        assert_eq!(mag.len(), n / 2 + 1);
        let argmax = (0..mag.len()).max_by(|&a, &b| mag[a].total_cmp(&mag[b])).unwrap();
        assert_eq!(argmax, 8);
        assert!((mag[8] - n as f64 / 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_rfft_frequencies() {
        assert_eq!(rfft_frequencies(8, 8000), vec![0.0, 1000.0, 2000.0, 3000.0, 4000.0]);
        assert_eq!(rfft_frequencies(0, 8000), Vec::<f64>::new());
    }
}
