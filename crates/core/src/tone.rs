//! Pure-tone analysis used by the marker-tone verification strategy.

use crate::dsp::{find_peaks_1d, hanning, rfft_frequencies, rfft_magnitude, FindPeaksOptions};

/// Frequency-domain metrics for validating a pure-tone candidate window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PureToneMetrics {
    pub detected_frequency: f64,
    pub overall_band_purity: f64,
    pub active_frame_ratio: f64,
    pub longest_active_run: usize,
    pub active_frame_mean_purity: f64,
}

/// `math.isclose`-style comparison: relative to the larger magnitude, or absolute.
pub(crate) fn is_close(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    (a - b).abs() <= (rel_tol * a.abs().max(b.abs())).max(abs_tol)
}

/// Index of the first maximum.
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (idx, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = idx;
        }
    }
    best
}

fn band_energy(spectrum: &[f64], freqs: &[f64], center: f64, half_width: f64) -> f64 {
    spectrum
        .iter()
        .zip(freqs)
        .filter(|(_, &f)| (f - center).abs() <= half_width)
        .map(|(&s, _)| s * s)
        .sum()
}

/// Return the dominant frequency if the audio is a pure tone, else `None`.
pub fn get_pure_tone_frequency(audio_data: &[f32], sample_rate: u32) -> Option<f64> {
    if audio_data.is_empty() {
        return None;
    }
    let samples: Vec<f64> = audio_data.iter().map(|&v| v as f64).collect();
    let magnitude = rfft_magnitude(&samples);
    let freqs = rfft_frequencies(audio_data.len(), sample_rate);

    let dominant_idx = argmax(&magnitude);
    let dominant_magnitude = magnitude[dominant_idx];
    if dominant_magnitude == 0.0 {
        return None;
    }
    let normalized: Vec<f32> = magnitude.iter().map(|&m| (m / dominant_magnitude) as f32).collect();

    let peaks = find_peaks_1d(
        &normalized,
        &FindPeaksOptions { height: None, distance: None, prominence: Some(0.05) },
    );

    let dominant_freq = freqs[dominant_idx];
    if peaks.len() == 1 && is_close(freqs[peaks[0]], dominant_freq, 0.01, 0.0) {
        Some(dominant_freq)
    } else {
        None
    }
}

/// Measure how strongly a candidate window behaves like a single pure tone.
pub fn analyze_pure_tone_candidate(
    audio_data: &[f32],
    sample_rate: u32,
    dominant_frequency: f64,
) -> PureToneMetrics {
    let mut metrics = PureToneMetrics {
        detected_frequency: 0.0,
        overall_band_purity: 0.0,
        active_frame_ratio: 0.0,
        longest_active_run: 0,
        active_frame_mean_purity: 0.0,
    };
    if audio_data.is_empty() {
        return metrics;
    }

    let target_band_hz = (dominant_frequency * 0.08).max(40.0);
    let target_lock_hz = (dominant_frequency * 0.04).max(20.0);

    let window = hanning(audio_data.len());
    let windowed: Vec<f64> = audio_data.iter().zip(&window).map(|(&a, &w)| a as f64 * w).collect();
    let spectrum = rfft_magnitude(&windowed);
    let freqs = rfft_frequencies(audio_data.len(), sample_rate);
    metrics.detected_frequency = freqs[argmax(&spectrum)];

    let total_energy: f64 = spectrum.iter().map(|s| s * s).sum();
    if total_energy == 0.0 {
        return metrics;
    }
    metrics.overall_band_purity =
        band_energy(&spectrum, &freqs, dominant_frequency, target_band_hz) / total_energy;

    let window_len = ((0.025 * sample_rate as f64).round() as usize).max(32);
    let hop = (window_len / 2).max(1);
    let frame_window = hanning(window_len);
    let frame_freqs = rfft_frequencies(window_len, sample_rate);

    let mut frame_count = 0usize;
    let mut active_frame_count = 0usize;
    let mut current_active_run = 0usize;
    let mut active_purity_sum = 0.0_f64;

    let mut start = 0;
    while start + window_len < audio_data.len() {
        let frame: Vec<f64> = audio_data[start..start + window_len]
            .iter()
            .zip(&frame_window)
            .map(|(&a, &w)| a as f64 * w)
            .collect();
        start += hop;

        let frame_spectrum = rfft_magnitude(&frame);
        let frame_energy: f64 = frame_spectrum.iter().map(|s| s * s).sum();
        if frame_energy == 0.0 {
            current_active_run = 0;
            continue;
        }

        frame_count += 1;
        let frame_dominant_frequency = frame_freqs[argmax(&frame_spectrum)];
        let frame_target_purity =
            band_energy(&frame_spectrum, &frame_freqs, dominant_frequency, target_band_hz) / frame_energy;

        let is_active = is_close(frame_dominant_frequency, dominant_frequency, 1e-9, target_lock_hz)
            && frame_target_purity >= 0.55;
        if is_active {
            active_frame_count += 1;
            current_active_run += 1;
            metrics.longest_active_run = metrics.longest_active_run.max(current_active_run);
            active_purity_sum += frame_target_purity;
        } else {
            current_active_run = 0;
        }
    }

    if frame_count > 0 {
        metrics.active_frame_ratio = active_frame_count as f64 / frame_count as f64;
    }
    if active_frame_count > 0 {
        metrics.active_frame_mean_purity = active_purity_sum / active_frame_count as f64;
    }
    metrics
}

/// Extract a fixed-length segment, padding with zeros when out of bounds.
pub fn extract_padded_segment(audio_data: &[f32], start: isize, length: usize) -> Vec<f32> {
    let len = audio_data.len() as isize;
    let stop = start + length as isize;
    let bounded_start = start.clamp(0, len) as usize;
    let bounded_stop = stop.clamp(0, len) as usize;

    let mut segment = vec![0.0_f32; length];
    if bounded_start < bounded_stop {
        let offset = (bounded_start as isize - start) as usize;
        segment[offset..offset + (bounded_stop - bounded_start)]
            .copy_from_slice(&audio_data[bounded_start..bounded_stop]);
    }
    segment
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency: f64, samples: usize, sample_rate: u32) -> Vec<f32> {
        (0..samples)
            .map(|i| (2.0 * std::f64::consts::PI * frequency * i as f64 / sample_rate as f64).sin() as f32)
            .collect()
    }

    #[test]
    fn test_extract_padded_segment() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(extract_padded_segment(&data, 1, 2), vec![2.0, 3.0]);
        assert_eq!(extract_padded_segment(&data, -2, 4), vec![0.0, 0.0, 1.0, 2.0]);
        assert_eq!(extract_padded_segment(&data, 2, 4), vec![3.0, 4.0, 0.0, 0.0]);
        assert_eq!(extract_padded_segment(&data, -1, 6), vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0]);
        assert_eq!(extract_padded_segment(&data, -5, 3), vec![0.0, 0.0, 0.0]);
        assert_eq!(extract_padded_segment(&data, 7, 2), vec![0.0, 0.0]);
    }

    #[test]
    fn test_is_close() {
        assert!(is_close(1040.0, 1000.0, 0.05, 0.0));
        assert!(!is_close(1100.0, 1000.0, 0.05, 0.0));
        assert!(is_close(1020.0, 1000.0, 1e-9, 20.0));
        assert!(!is_close(1021.0, 1000.0, 1e-9, 20.0));
    }

    #[test]
    fn test_pure_tone_frequency_of_sine() {
        let freq = get_pure_tone_frequency(&sine(1000.0, 1840, 8000), 8000).unwrap();
        assert!(is_close(freq, 1000.0, 0.01, 0.0), "got {freq}");
    }

    #[test]
    fn test_pure_tone_frequency_rejects_two_tones_and_silence() {
        let mixed: Vec<f32> = sine(1000.0, 1840, 8000)
            .iter()
            .zip(sine(2500.0, 1840, 8000))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(get_pure_tone_frequency(&mixed, 8000), None);
        assert_eq!(get_pure_tone_frequency(&[0.0; 800], 8000), None);
        assert_eq!(get_pure_tone_frequency(&[], 8000), None);
    }

    #[test]
    fn test_analyze_clean_tone_is_fully_active() {
        let metrics = analyze_pure_tone_candidate(&sine(1040.0, 1827, 8000), 8000, 1040.0);
        assert!(is_close(metrics.detected_frequency, 1040.0, 0.01, 0.0));
        assert!(metrics.overall_band_purity > 0.99, "{metrics:?}");
        assert_eq!(metrics.active_frame_ratio, 1.0);
        // window 200, hop 100: frames start at 0, 100, ..., 1600
        assert_eq!(metrics.longest_active_run, 17);
        assert!(metrics.active_frame_mean_purity > 0.99);
    }

    #[test]
    fn test_analyze_empty_and_silent_segments() {
        let empty = analyze_pure_tone_candidate(&[], 8000, 1040.0);
        assert_eq!(empty.detected_frequency, 0.0);
        assert_eq!(empty.overall_band_purity, 0.0);

        let silent = analyze_pure_tone_candidate(&[0.0; 1827], 8000, 1040.0);
        assert_eq!(silent.overall_band_purity, 0.0);
        assert_eq!(silent.active_frame_ratio, 0.0);
        assert_eq!(silent.longest_active_run, 0);
    }
}
