use std::path::Path;

use audio_pattern_detector_core::detector::{analyze_tone_candidate_context, marker_tone_accepts};
use audio_pattern_detector_core::dsp::hanning;
use audio_pattern_detector_core::{
    AudioClip, AudioPatternDetector, DetectorOptions, MarkerToneParams, Strategy, DEFAULT_TARGET_SAMPLE_RATE,
};

const RTHK_BEEP_PATTERN: &str = "../../sample_audios/clips/rthk_beep.apd.toml";
const RTHK_BEEP_NAME: &str = "rthk_beep";
const HARMONIC_STACK_FUNDAMENTAL: f64 = 260.0;
const HARMONIC_STACK_AMPLITUDES: [f32; 5] = [0.50, 0.35, 0.30, 0.28, 0.22];
const SWEEP_START_FREQUENCY: f64 = 920.0;
const SWEEP_END_FREQUENCY: f64 = 1160.0;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

// The builders mirror the float32 arithmetic of the numpy originals.

fn active_envelope(active_samples: usize) -> Vec<f32> {
    hanning(active_samples).into_iter().map(|v| v as f32).collect()
}

fn sample_times(samples: usize, sample_rate: u32) -> Vec<f32> {
    (0..samples).map(|i| i as f32 / sample_rate as f32).collect()
}

fn build_clean_candidate(length: usize, sample_rate: u32, frequency: f64) -> Vec<f32> {
    let envelope = active_envelope(length);
    let angular_frequency = (TWO_PI * frequency) as f32;
    sample_times(length, sample_rate)
        .iter()
        .zip(&envelope)
        .map(|(&t, &e)| 0.9_f32 * (angular_frequency * t).sin() * e)
        .collect()
}

fn build_harmonic_stack_candidate(length: usize, sample_rate: u32) -> Vec<f32> {
    let envelope = active_envelope(length);
    let mut signal: Vec<f32> = sample_times(length, sample_rate)
        .iter()
        .zip(&envelope)
        .map(|(&t, &e)| {
            let mut harmonic_stack = 0.0_f32;
            for (harmonic, amplitude) in HARMONIC_STACK_AMPLITUDES.iter().enumerate() {
                let angular_frequency = (TWO_PI * HARMONIC_STACK_FUNDAMENTAL * (harmonic + 1) as f64) as f32;
                harmonic_stack += amplitude * (angular_frequency * t).sin();
            }
            harmonic_stack * e
        })
        .collect();
    let peak = signal.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
    signal.iter_mut().for_each(|v| *v /= peak);
    signal
}

fn build_swept_candidate(length: usize, sample_rate: u32) -> Vec<f32> {
    let envelope = active_envelope(length);
    let step = (SWEEP_END_FREQUENCY - SWEEP_START_FREQUENCY) / (length - 1) as f64;
    let two_pi = TWO_PI as f32;
    let mut cumulative_frequency = 0.0_f32;
    (0..length)
        .map(|i| {
            let instantaneous_frequency = if i == length - 1 {
                SWEEP_END_FREQUENCY
            } else {
                SWEEP_START_FREQUENCY + i as f64 * step
            };
            cumulative_frequency += instantaneous_frequency as f32;
            let phase = two_pi * cumulative_frequency / sample_rate as f32;
            0.9_f32 * phase.sin() * envelope[i]
        })
        .collect()
}

/// Equivalent of the detector's private marker-tone verifier.
fn run_verify_marker_tone(params: &MarkerToneParams, audio_section: &[f32], dominant_frequency: f64) -> bool {
    // peak = len-1 and clip_length = len -> match_start = 0, so the
    // entire audio_section is used as the matched segment.
    let (metrics, left_metrics, right_metrics) = analyze_tone_candidate_context(
        audio_section,
        audio_section.len() - 1,
        audio_section.len(),
        dominant_frequency,
        DEFAULT_TARGET_SAMPLE_RATE,
    );
    marker_tone_accepts(
        dominant_frequency,
        &params.thresholds,
        &metrics,
        &left_metrics,
        &right_metrics,
    )
}

#[test]
fn test_marker_tone_verifier_rejects_harmonic_and_swept_false_positives() {
    assert!(
        Path::new(RTHK_BEEP_PATTERN).exists(),
        "Pattern file {RTHK_BEEP_PATTERN} not found"
    );

    let pattern_clip = AudioClip::from_audio_file(RTHK_BEEP_PATTERN, DEFAULT_TARGET_SAMPLE_RATE).unwrap();
    let Some(Strategy::MarkerTone(params)) = pattern_clip.strategy.clone() else {
        panic!("{RTHK_BEEP_PATTERN} should declare the marker_tone strategy");
    };
    let dominant_frequency = params.dominant_frequency_hz.expect("dominant_frequency_hz should be set");
    let candidate_length = pattern_clip.audio.len();

    // The detector verifies this clip's candidates with the same parameters.
    let detector = AudioPatternDetector::new(vec![pattern_clip], DetectorOptions::default()).unwrap();
    assert!(detector.uses_marker_tone(RTHK_BEEP_NAME));

    let clean_candidate = build_clean_candidate(candidate_length, DEFAULT_TARGET_SAMPLE_RATE, dominant_frequency);
    let harmonic_candidate = build_harmonic_stack_candidate(candidate_length, DEFAULT_TARGET_SAMPLE_RATE);
    let swept_candidate = build_swept_candidate(candidate_length, DEFAULT_TARGET_SAMPLE_RATE);

    let verification_results = [
        run_verify_marker_tone(&params, &clean_candidate, dominant_frequency),
        run_verify_marker_tone(&params, &harmonic_candidate, dominant_frequency),
        run_verify_marker_tone(&params, &swept_candidate, dominant_frequency),
    ];

    assert_eq!(verification_results, [true, false, false]);
}
