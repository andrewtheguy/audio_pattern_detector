//! Port of `tests/test_audio_utils.py`: WAV reading/writing, resampling, ffmpeg availability.
//!
//! Python's `load_wave_file(path, sr)` (native WAV decode, ffmpeg for anything
//! else, resampled to `sr`) maps to `AudioClip::from_audio_file(path, sr).audio`.

use std::path::Path;

use audio_pattern_detector_core::ffmpeg::is_ffmpeg_available;
use audio_pattern_detector_core::stream::resample_audio;
use audio_pattern_detector_core::wav::{load_wav_file, write_wav_file};
use audio_pattern_detector_core::{AudioClip, Result};
use tempfile::TempDir;

const CBS_NEWS_CLIP: &str = "../../sample_audios/clips/cbs_news.wav";
const NONEXISTENT_WAV: &str = "nonexistent_file.wav";
const TEMP_WAV_NAME: &str = "temp.wav";

fn load_wave_file(path: impl AsRef<Path>, expected_sample_rate: u32) -> Result<Vec<f32>> {
    AudioClip::from_audio_file(path, expected_sample_rate).map(|clip| clip.audio)
}

fn max_abs(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
}

/// `np.testing.assert_array_almost_equal(expected, actual, decimal=decimal)`.
fn assert_array_almost_equal(expected: &[f32], actual: &[f32], decimal: i32) {
    assert_eq!(expected.len(), actual.len(), "array lengths differ");
    let tolerance = 1.5 * 10f64.powi(-decimal);
    for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
        let diff = (*e as f64 - *a as f64).abs();
        assert!(diff < tolerance, "index {i}: expected {e}, got {a} (diff {diff} >= {tolerance})");
    }
}

/// `np.sin(2 * np.pi * frequency * np.arange(samples) / sample_rate).astype(np.float32)`.
fn sine_arange(frequency: f64, samples: usize, sample_rate: u32) -> Vec<f32> {
    (0..samples)
        .map(|i| (2.0 * std::f64::consts::PI * frequency * i as f64 / sample_rate as f64).sin() as f32)
        .collect()
}

/// 16-bit PCM WAV with interleaved stereo frames.
fn stereo_wav_bytes(left: &[i16], right: &[i16], sample_rate: u32) -> Vec<u8> {
    assert_eq!(left.len(), right.len());
    let data_len = (left.len() * 4) as u32;
    let mut bytes = Vec::with_capacity(44 + left.len() * 4);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&2u16.to_le_bytes()); // stereo
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 4).to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for (l, r) in left.iter().zip(right) {
        bytes.extend_from_slice(&l.to_le_bytes());
        bytes.extend_from_slice(&r.to_le_bytes());
    }
    bytes
}

/// `np.corrcoef(a, b)[0, 1]`.
fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mean_b = b.iter().map(|v| *v as f64).sum::<f64>() / n;
    let (mut cov, mut var_a, mut var_b) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (*x as f64 - mean_a, *y as f64 - mean_b);
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    cov / (var_a * var_b).sqrt()
}

mod write_wav_file {
    use super::*;

    #[test]
    fn test_write_and_read_roundtrip() {
        let sample_rate = 8000u32;
        let duration = 1.0f64;
        let n = (sample_rate as f64 * duration) as usize;
        // np.linspace(0, duration, n) includes the endpoint.
        let audio_data: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f64 * duration / (n - 1) as f64) as f32;
                (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t as f64).sin()) as f32
            })
            .collect();

        let dir = TempDir::new().unwrap();
        let temp_path = dir.path().join(TEMP_WAV_NAME);
        write_wav_file(&temp_path, &audio_data, sample_rate).unwrap();

        assert!(temp_path.exists());
        assert!(std::fs::metadata(&temp_path).unwrap().len() > 0);

        let loaded_audio = load_wave_file(&temp_path, sample_rate).unwrap();
        assert_array_almost_equal(&audio_data, &loaded_audio, 4);
    }

    #[test]
    fn test_write_different_sample_rates() {
        for sample_rate in [8000u32, 16000, 44100] {
            let audio_data = vec![0.0f32; sample_rate as usize]; // 1 second of silence

            let dir = TempDir::new().unwrap();
            let temp_path = dir.path().join(TEMP_WAV_NAME);
            write_wav_file(&temp_path, &audio_data, sample_rate).unwrap();
            assert!(temp_path.exists());

            let loaded = load_wave_file(&temp_path, sample_rate).unwrap();
            assert_eq!(loaded.len(), sample_rate as usize, "sample rate {sample_rate}");
        }
    }
}

mod load_wave_file {
    use super::*;

    #[test]
    fn test_load_existing_wav_file() {
        let audio = load_wave_file(CBS_NEWS_CLIP, 8000).unwrap();
        assert!(!audio.is_empty());
        // Check normalized range
        assert!(max_abs(&audio) <= 1.0);
    }

    #[test]
    fn test_load_with_different_sample_rate_resamples() {
        // Load at 8kHz (original rate)
        let audio_8k = load_wave_file(CBS_NEWS_CLIP, 8000).unwrap();
        // Load at 16kHz (should resample)
        let audio_16k = load_wave_file(CBS_NEWS_CLIP, 16000).unwrap();
        // 16kHz version should have approximately twice as many samples
        let expected = (audio_8k.len() * 2) as f64;
        assert!(
            (audio_16k.len() as f64 - expected).abs() <= 0.01 * expected,
            "16k length {} vs expected ~{expected}",
            audio_16k.len()
        );
    }

    #[test]
    fn test_load_nonexistent_file_raises() {
        assert!(load_wave_file(NONEXISTENT_WAV, 8000).is_err());
    }

    #[test]
    fn test_load_stereo_file_converts_to_mono() {
        let sample_rate = 8000u32;
        let duration_seconds = 1;
        let num_samples = (sample_rate * duration_seconds) as usize;
        // Different values for left and right channels to verify mixing.
        let audio_left = vec![16384i16; num_samples]; // ~0.5
        let audio_right = vec![-16384i16; num_samples]; // ~-0.5

        let dir = TempDir::new().unwrap();
        let temp_path = dir.path().join(TEMP_WAV_NAME);
        std::fs::write(&temp_path, stereo_wav_bytes(&audio_left, &audio_right, sample_rate)).unwrap();

        // Should load successfully (stereo converted to mono)
        let audio = load_wave_file(&temp_path, sample_rate).unwrap();
        assert_eq!(audio.len(), num_samples);
        // Mono conversion should average left and right channels (0.5 + -0.5) / 2 ≈ 0
        assert!(max_abs(&audio) < 0.1);
    }
}

mod round_trip {
    use super::*;

    #[test]
    fn test_preserves_audio_content() {
        let sample_rate = 8000u32;
        let audio_data: Vec<f32> = vec![0.0, 0.5, -0.5, 0.99, -0.99, 0.25, -0.25];

        let dir = TempDir::new().unwrap();
        let temp_path = dir.path().join(TEMP_WAV_NAME);
        write_wav_file(&temp_path, &audio_data, sample_rate).unwrap();
        let loaded = load_wave_file(&temp_path, sample_rate).unwrap();

        // Values should be close (16-bit quantization introduces small errors)
        assert_array_almost_equal(&audio_data, &loaded, 4);
    }

    #[test]
    fn test_load_sample_file_and_rewrite() {
        let sample_rate = 8000u32;
        let original = load_wave_file(CBS_NEWS_CLIP, sample_rate).unwrap();

        let dir = TempDir::new().unwrap();
        let temp_path = dir.path().join(TEMP_WAV_NAME);
        write_wav_file(&temp_path, &original, sample_rate).unwrap();
        let reloaded = load_wave_file(&temp_path, sample_rate).unwrap();

        // Should be identical
        assert_array_almost_equal(&original, &reloaded, 5);
    }
}

mod audio_utilities {
    use super::*;

    #[test]
    fn test_is_ffmpeg_available_returns_bool() {
        let _result: bool = is_ffmpeg_available();
    }

    // The Python test also reset and inspected the private cache variable;
    // only the observable "repeated calls agree" half exists in Rust.
    #[test]
    fn test_is_ffmpeg_available_cached() {
        let first_call = is_ffmpeg_available();
        let second_call = is_ffmpeg_available();
        assert_eq!(first_call, second_call);
    }

    #[test]
    fn test_load_wav_file_basic() {
        let (audio, sample_rate) = load_wav_file(CBS_NEWS_CLIP).unwrap();
        assert_eq!(sample_rate, 8000);
        assert!(!audio.is_empty());
        // Check normalized range
        assert!(max_abs(&audio) <= 1.0);
    }

    #[test]
    fn test_load_wav_file_int16() {
        // All our sample files are 16-bit; should come back normalized.
        let (audio, _sample_rate) = load_wav_file(CBS_NEWS_CLIP).unwrap();
        assert!(max_abs(&audio) <= 1.0);
    }

    #[test]
    fn test_load_wav_file_nonexistent() {
        let err = load_wav_file(NONEXISTENT_WAV).expect_err("nonexistent file should fail");
        let message = err.to_string();
        assert!(message.contains("Failed to read"), "unexpected error message: {message}");
    }

    #[test]
    fn test_resample_audio_same_rate() {
        let audio: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let result = resample_audio(audio.clone(), 8000, 8000);
        assert_eq!(result, audio);
    }

    #[test]
    fn test_resample_audio_downsample() {
        // 1 second of audio at 16kHz (16000 samples)
        let audio = sine_arange(440.0, 16000, 16000);
        let result = resample_audio(audio, 16000, 8000);
        // Should have 8000 samples (1 second at 8kHz)
        assert_eq!(result.len(), 8000);
    }

    #[test]
    fn test_resample_audio_upsample() {
        // 1 second of audio at 8kHz (8000 samples)
        let audio = sine_arange(440.0, 8000, 8000);
        let result = resample_audio(audio, 8000, 16000);
        // Should have 16000 samples (1 second at 16kHz)
        assert_eq!(result.len(), 16000);
    }

    #[test]
    fn test_resample_audio_preserves_frequency() {
        // 440Hz sine wave at 16kHz, 100ms
        let freq = 440.0;
        let duration = 0.1f64;
        let orig_sr = 16000u32;
        let target_sr = 8000u32;
        let audio = sine_arange(freq, (orig_sr as f64 * duration) as usize, orig_sr);

        let resampled = resample_audio(audio, orig_sr, target_sr);

        // Reference at target sample rate
        let reference = sine_arange(freq, (target_sr as f64 * duration) as usize, target_sr);

        // Should be similar (allow some tolerance due to resampling artifacts)
        assert_eq!(resampled.len(), reference.len());
        let correlation = pearson(&resampled, &reference);
        assert!(correlation > 0.99, "Correlation too low: {correlation}");
    }
}
