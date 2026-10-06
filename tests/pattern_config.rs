//! Port of `tests/test_pattern_config.py`: loading `.apd.toml` pattern configs.

use std::path::{Path, PathBuf};

use audio_pattern_detector::pattern_config::{load_apd_file, PatternConfig};
use audio_pattern_detector::{MarkerToneParams, MarkerToneThresholds, Strategy, DEFAULT_TARGET_SAMPLE_RATE};
use base64::Engine;
use tempfile::TempDir;

const DEFAULT_APD_NAME: &str = "clip.apd.toml";
const INLINE_APD_NAME: &str = "inline.apd.toml";
const BASE64_WRAP_WIDTH: usize = 76;

fn write_toml(dir: &TempDir, body: &str, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, body).unwrap();
    path
}

/// 16-bit mono PCM WAV of a unit sine, quantised exactly like the Python helper
/// (`int(sample * 32767)`, i.e. truncation toward zero).
fn sine_wav_bytes(frequency_hz: f64, duration_seconds: f64, sample_rate: u32) -> Vec<u8> {
    let n = (duration_seconds * sample_rate as f64).round() as usize;
    let data_len = (n * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + n * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..n {
        let value = (2.0 * std::f64::consts::PI * frequency_hz * i as f64 / sample_rate as f64).sin();
        let sample = (value.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn load(path: &Path) -> PatternConfig {
    load_apd_file(path, DEFAULT_TARGET_SAMPLE_RATE).unwrap()
}

fn marker_tone_params(config: &PatternConfig) -> &MarkerToneParams {
    let Strategy::MarkerTone(params) = &config.strategy;
    params
}

fn max_abs(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
}

/// Load `body` as an `.apd.toml` and assert it is rejected with a message containing `expected`.
fn assert_rejected(body: &str, expected: &str) {
    let dir = TempDir::new().unwrap();
    let path = write_toml(&dir, body, DEFAULT_APD_NAME);
    let err = load_apd_file(&path, DEFAULT_TARGET_SAMPLE_RATE).expect_err("config should be rejected");
    let message = err.to_string();
    assert!(message.contains(expected), "expected {expected:?} in error message, got: {message}");
}

#[test]
fn test_sine_source_round_trip() {
    let body = r#"[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1
amplitude = 1.0

[verification]
strategy = "marker_tone"
"#;
    let dir = TempDir::new().unwrap();
    let path = write_toml(&dir, body, DEFAULT_APD_NAME);
    let sr = DEFAULT_TARGET_SAMPLE_RATE;
    let config = load_apd_file(&path, sr).unwrap();

    assert_eq!(config.strategy.name(), "marker_tone");
    assert_eq!(config.audio.len(), (0.1 * sr as f64).round() as usize);
    let peak = max_abs(&config.audio) as f64;
    assert!((peak - 1.0).abs() <= 1e-3, "peak amplitude {peak}");
    // Sine source auto-populates dominant_frequency_hz from the declared frequency,
    // and no thresholds were provided.
    assert_eq!(
        config.strategy,
        Strategy::MarkerTone(MarkerToneParams {
            dominant_frequency_hz: Some(1040.0),
            thresholds: MarkerToneThresholds::default(),
        })
    );
}

#[test]
fn test_sine_source_with_thresholds_and_explicit_dominant_frequency() {
    let body = r#"[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1

[verification]
strategy = "marker_tone"
dominant_frequency_hz = 1041.5
minimum_band_purity = 0.72
minimum_active_frame_ratio = 0.70
minimum_longest_active_run = 7
minimum_active_frame_mean_purity = 0.77
maximum_min_flank_purity = 0.02
maximum_max_flank_purity = 0.14
"#;
    let dir = TempDir::new().unwrap();
    let path = write_toml(&dir, body, DEFAULT_APD_NAME);
    let config = load(&path);

    let params = marker_tone_params(&config);
    assert_eq!(params.dominant_frequency_hz, Some(1041.5));
    assert_eq!(
        params.thresholds,
        MarkerToneThresholds {
            minimum_band_purity: Some(0.72),
            minimum_active_frame_ratio: Some(0.70),
            minimum_longest_active_run: Some(7),
            minimum_active_frame_mean_purity: Some(0.77),
            maximum_min_flank_purity: Some(0.02),
            maximum_max_flank_purity: Some(0.14),
        }
    );
}

#[test]
fn test_wav_base64_round_trip() {
    let sr = DEFAULT_TARGET_SAMPLE_RATE;
    let freq = 1040.0f64;
    let dur = 0.1f64;
    let encoded = b64(&sine_wav_bytes(freq, dur, sr));
    let body = format!(
        r#"[clip]
source = "wav_base64"
data = "{encoded}"

[verification]
strategy = "marker_tone"
dominant_frequency_hz = {freq:?}
"#
    );
    let dir = TempDir::new().unwrap();
    let path = write_toml(&dir, &body, DEFAULT_APD_NAME);
    let config = load_apd_file(&path, sr).unwrap();

    let n = (dur * sr as f64).round() as usize;
    let expected: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
        .collect();
    assert_eq!(config.audio.len(), n);
    // int16 round-trip introduces ~1.5e-4 quantisation error, well below 1e-3.
    let max_diff = config
        .audio
        .iter()
        .zip(&expected)
        .fold(0.0f32, |acc, (a, e)| acc.max((a - e).abs()));
    assert!(max_diff < 1e-3, "max abs difference {max_diff}");
    assert_eq!(marker_tone_params(&config).dominant_frequency_hz, Some(freq));
}

// [clip].data may span multiple lines via TOML triple-quoted strings.
#[test]
fn test_wav_base64_accepts_multiline_string() {
    let sr = DEFAULT_TARGET_SAMPLE_RATE;
    let encoded = b64(&sine_wav_bytes(1040.0, 0.05, sr));
    let wrapped = encoded
        .as_bytes()
        .chunks(BASE64_WRAP_WIDTH)
        .map(|chunk| std::str::from_utf8(chunk).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    let body = format!(
        r#"[clip]
source = "wav_base64"
data = """
{wrapped}
"""

[verification]
strategy = "marker_tone"
dominant_frequency_hz = 1040.0
"#
    );
    let inline_body = format!(
        r#"[clip]
source = "wav_base64"
data = "{encoded}"

[verification]
strategy = "marker_tone"
dominant_frequency_hz = 1040.0
"#
    );
    let dir = TempDir::new().unwrap();
    let path = write_toml(&dir, &body, DEFAULT_APD_NAME);
    let inline_path = write_toml(&dir, &inline_body, INLINE_APD_NAME);

    let multiline = load_apd_file(&path, sr).unwrap();
    let inline = load_apd_file(&inline_path, sr).unwrap();
    assert!(!inline.audio.is_empty());
    assert_eq!(multiline.audio, inline.audio);
}

#[test]
fn test_wav_base64_resamples_to_target() {
    // WAV recorded at 16 kHz, loader must resample to the requested 8 kHz target.
    let source_sr = 16000;
    let target_sr = 8000;
    let encoded = b64(&sine_wav_bytes(1000.0, 0.1, source_sr));
    let body = format!(
        r#"[clip]
source = "wav_base64"
data = "{encoded}"

[verification]
strategy = "marker_tone"
dominant_frequency_hz = 1000.0
"#
    );
    let dir = TempDir::new().unwrap();
    let path = write_toml(&dir, &body, DEFAULT_APD_NAME);
    let config = load_apd_file(&path, target_sr).unwrap();

    assert_eq!(config.audio.len(), (0.1 * target_sr as f64).round() as usize);
}

#[test]
fn test_top_level_strategy_is_rejected() {
    let body = r#"strategy = "marker_tone"

[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1

[verification]
strategy = "marker_tone"
"#;
    assert_rejected(body, "unknown top-level field");
}

#[test]
fn test_legacy_generator_section_is_rejected() {
    let body = r#"strategy = "marker_tone"

[generator]
type = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1
"#;
    assert_rejected(body, "unknown top-level field");
}

#[test]
fn test_unknown_clip_source_is_rejected() {
    let body = r#"[clip]
source = "square"
frequency_hz = 1040.0

[verification]
strategy = "marker_tone"
"#;
    assert_rejected(body, "unknown [clip].source 'square'");
}

#[test]
fn test_unknown_strategy_is_rejected() {
    let body = r#"[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1

[verification]
strategy = "pure_tone"
"#;
    assert_rejected(body, "unknown strategy 'pure_tone'");
}

#[test]
fn test_unknown_clip_field_for_sine_is_rejected() {
    let body = r#"[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1
data = "abc"

[verification]
strategy = "marker_tone"
"#;
    assert_rejected(body, "unknown [clip] field");
}

#[test]
fn test_unknown_clip_field_for_wav_base64_is_rejected() {
    let body = r#"[clip]
source = "wav_base64"
data = "AAAA"
frequency_hz = 1040.0

[verification]
strategy = "marker_tone"
"#;
    assert_rejected(body, "unknown [clip] field");
}

#[test]
fn test_unknown_verification_field_is_rejected() {
    let body = r#"[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1

[verification]
strategy = "marker_tone"
not_a_real_threshold = 0.5
"#;
    assert_rejected(body, "unknown [verification] field");
}

#[test]
fn test_invalid_base64_is_rejected() {
    let body = r#"[clip]
source = "wav_base64"
data = "not!valid!base64!"

[verification]
strategy = "marker_tone"
"#;
    assert_rejected(body, "invalid base64");
}

#[test]
fn test_missing_clip_section_is_rejected() {
    let body = r#"[verification]
strategy = "marker_tone"
"#;
    assert_rejected(body, "missing required field 'clip'");
}

#[test]
fn test_missing_verification_section_is_rejected() {
    let body = r#"[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1
"#;
    assert_rejected(body, "missing required field 'verification'");
}

#[test]
fn test_non_finite_numbers_are_rejected() {
    // (field, TOML literal, how the value is displayed)
    for (field, value, shown) in [
        ("duration_seconds", "inf", "inf"),
        ("duration_seconds", "nan", "NaN"),
        ("frequency_hz", "inf", "inf"),
    ] {
        let mut clip = [("frequency_hz", "1040.0"), ("duration_seconds", "0.23")];
        clip.iter_mut().filter(|(key, _)| *key == field).for_each(|entry| entry.1 = value);
        let body = format!(
            "[clip]\nsource = \"sine\"\n{} = {}\n{} = {}\n\n[verification]\nstrategy = \"marker_tone\"\n",
            clip[0].0, clip[0].1, clip[1].0, clip[1].1
        );
        assert_rejected(&body, &format!("'{field}' must be a finite number, got {shown}"));
    }
    assert_rejected(
        "[clip]\nsource = \"sine\"\nfrequency_hz = 1040.0\nduration_seconds = 0.23\n\n\
         [verification]\nstrategy = \"marker_tone\"\nminimum_band_purity = nan\n",
        "'minimum_band_purity' must be a finite number, got NaN",
    );
}

#[test]
fn test_excessive_sine_duration_is_rejected() {
    assert_rejected(
        "[clip]\nsource = \"sine\"\nfrequency_hz = 1040.0\nduration_seconds = 7200.5\n\n\
         [verification]\nstrategy = \"marker_tone\"\n",
        "duration_seconds must be at most 3600, got 7200.5",
    );
}
