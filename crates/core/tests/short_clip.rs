//! Tests for short clip detection through the normal correlation path.
//!
//! Short clips (< 0.5s) go through the normal path with a 0-100% window,
//! not the marker tone verification path.

mod common;

use std::io::Cursor;

use audio_pattern_detector_core::detector::SHORT_CLIP_DURATION_THRESHOLD;
use audio_pattern_detector_core::dsp::hanning;
use audio_pattern_detector_core::stream::F32leSource;
use audio_pattern_detector_core::tone::get_pure_tone_frequency;
use audio_pattern_detector_core::{
    AudioClip, AudioPatternDetector, AudioStream, DetectorOptions, MarkerToneParams, MarkerToneThresholds,
    Strategy,
};
use common::{clip_from_samples, concat, Rng, SR};

const CHIRP_DURATION: f64 = 0.1; // seconds, well below the 0.5s threshold
const CHIRP_START_FREQUENCY: f64 = 400.0;
const CHIRP_END_FREQUENCY: f64 = 1200.0;

const TONE_DURATION: f64 = 0.125;
const TONE_FREQUENCY: f64 = 1000.0;

const NOISE_SEED: u64 = 42;

/// Linear chirp with a Hann envelope. Mirrors the float32 arithmetic of the
/// numpy original.
fn make_chirp(duration: f64, f0: f64, f1: f64) -> Vec<f32> {
    let n = (duration * SR as f64) as usize;
    let window = hanning(n);
    let two_pi = (2.0 * std::f64::consts::PI) as f32;
    let (f0, sweep, denominator) = (f0 as f32, (f1 - f0) as f32, (2.0 * duration) as f32);
    (0..n)
        .map(|i| {
            let t = i as f32 / SR as f32;
            let phase = two_pi * (f0 * t + sweep * (t * t) / denominator);
            ((0.8_f32 * phase.sin()) as f64 * window[i]) as f32
        })
        .collect()
}

/// Unwindowed sine at 0.9 amplitude.
fn make_tone(duration: f64, frequency: f64) -> Vec<f32> {
    let n = (duration * SR as f64) as usize;
    let angular_frequency = (2.0 * std::f64::consts::PI * frequency) as f32;
    (0..n)
        .map(|i| 0.9_f32 * (angular_frequency * (i as f32 / SR as f32)).sin())
        .collect()
}

/// Audio stream over raw float32 PCM bytes (no WAV header).
fn audio_stream_from_samples(name: &str, audio: &[f32]) -> AudioStream<'static> {
    let raw: Vec<u8> = audio.iter().flat_map(|sample| sample.to_le_bytes()).collect();
    AudioStream::new(name, F32leSource::new(Cursor::new(raw)), SR)
}

fn detector_for(clip: AudioClip) -> AudioPatternDetector {
    AudioPatternDetector::new(vec![clip], DetectorOptions::default()).unwrap()
}

// --- Tests ---

// Even if FFT analysis might find a dominant frequency, clips without
// strategy metadata still use the normal path.
#[test]
fn test_short_chirp_does_not_trigger_marker_tone_path() {
    let chirp = make_chirp(CHIRP_DURATION, CHIRP_START_FREQUENCY, CHIRP_END_FREQUENCY);
    let detector = detector_for(clip_from_samples("my_chirp", &chirp));

    assert!(!detector.uses_marker_tone("my_chirp"));
}

// Sanity check: a chirp with duration just under the threshold is a short clip.
#[test]
fn test_make_chirp_produces_sub_threshold_length() {
    let chirp = make_chirp(
        SHORT_CLIP_DURATION_THRESHOLD - 0.01,
        CHIRP_START_FREQUENCY,
        CHIRP_END_FREQUENCY,
    );
    assert!((chirp.len() as f64 / SR as f64) < SHORT_CLIP_DURATION_THRESHOLD);
}

#[test]
fn test_short_chirp_detected_in_audio() {
    let chirp = make_chirp(CHIRP_DURATION, CHIRP_START_FREQUENCY, CHIRP_END_FREQUENCY);

    // Test audio: 2s silence, chirp, 2s silence, chirp, 2s silence
    let silence = vec![0.0_f32; 2 * SR as usize];
    let test_audio = concat(&[&silence, &chirp, &silence, &chirp, &silence]);

    let detector = detector_for(clip_from_samples("test_chirp", &chirp));

    let mut stream = audio_stream_from_samples("test_audio", &test_audio);
    let (peak_times, _total_time) = detector.find_clip_in_audio(&mut stream, None, true).unwrap();

    let peak_times = peak_times.expect("peak_times should not be None");
    let mut matches = peak_times.get("test_chirp").expect("test_chirp key should exist").clone();
    matches.sort_by(f64::total_cmp);
    assert_eq!(matches.len(), 2, "matches: {matches:?}");

    // Chirps placed at 2.0s and 4.1s (2 + 0.1 + 2 = 4.1)
    let expected_positions = [2.0 + CHIRP_DURATION, 2.0 + CHIRP_DURATION + 2.0 + CHIRP_DURATION];
    for (actual, expected) in matches.iter().zip(expected_positions) {
        assert!((actual - expected).abs() < 0.15, "Expected ~{expected}s, got {actual}s");
    }
}

#[test]
fn test_short_chirp_no_false_positives_in_noise() {
    let chirp = make_chirp(CHIRP_DURATION, CHIRP_START_FREQUENCY, CHIRP_END_FREQUENCY);

    let noise = Rng::new(NOISE_SEED).noise(6 * SR as usize, 0.05);

    let detector = detector_for(clip_from_samples("test_chirp", &chirp));

    let mut stream = audio_stream_from_samples("noise_audio", &noise);
    let (peak_times, _) = detector.find_clip_in_audio(&mut stream, None, true).unwrap();

    let peak_times = peak_times.expect("peak_times should not be None");
    assert_eq!(peak_times.get("test_chirp").cloned().unwrap_or_default(), Vec::<f64>::new());
}

// A clip with the marker_tone strategy routes to the tone verifier path.
#[test]
fn test_marker_tone_strategy_triggers_tone_path() {
    let tone = make_tone(TONE_DURATION, TONE_FREQUENCY);

    let clip = AudioClip::new("my_marker", tone, SR).with_strategy(Strategy::MarkerTone(MarkerToneParams {
        dominant_frequency_hz: Some(TONE_FREQUENCY),
        thresholds: MarkerToneThresholds::default(),
    }));
    let detector = detector_for(clip);

    assert!(
        detector.uses_marker_tone("my_marker"),
        "the marker_tone strategy should register a dominant frequency"
    );
}

// A tone clip without the marker_tone strategy must NOT trigger the tone path.
#[test]
fn test_tone_clip_without_strategy_uses_normal_path() {
    let tone = make_tone(TONE_DURATION, TONE_FREQUENCY);

    // It IS a pure tone (audio content).
    assert!(get_pure_tone_frequency(&tone, SR).is_some());

    // Without strategy metadata, dispatch defaults to the normal path.
    let detector = detector_for(clip_from_samples("other_tone", &tone));

    assert!(
        !detector.uses_marker_tone("other_tone"),
        "Clips without the marker_tone strategy should not route to the tone path"
    );
}
