//! Port of `tests/test_real_data_regressions.py`: marker-tone regressions on real radio captures.

use std::path::Path;

use audio_pattern_detector_core::{match_pattern, MatchOptions};

/// `(audio file, expected timestamps in seconds)`.
type Case = (&'static str, &'static [f64]);

const RTHK_BEEP_PATTERN: &str = "../../sample_audios/clips/rthk_beep.apd.toml";
const RADIO903_BEEP_PATTERN: &str = "../../sample_audios/clips/903_beep.apd.toml";
const RADIO881_BEEP_PATTERN: &str = "../../sample_audios/clips/881_beep.apd.toml";

const RTHK_BEEP_CLIP_NAME: &str = "rthk_beep";
const RADIO903_BEEP_CLIP_NAME: &str = "903_beep";
const RADIO881_BEEP_CLIP_NAME: &str = "881_beep";

const RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_1: &str =
    "../../sample_audios/regressions/rthk_beep_stray_clips_v2/tp_09-10_beep1.wav";
const RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_2: &str =
    "../../sample_audios/regressions/rthk_beep_stray_clips_v2/tp_09-10_beep2.wav";
const RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_3: &str =
    "../../sample_audios/regressions/rthk_beep_stray_clips_v2/tp_09-10_beep3.wav";

const RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_1: &str =
    "../../sample_audios/regressions/rthk_beep_stray_clips_v2/v2_10-11_20m21s.wav";
const RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_2: &str =
    "../../sample_audios/regressions/rthk_beep_stray_clips_v2/v2_10-11_50m40s.wav";
const RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_3: &str =
    "../../sample_audios/regressions/rthk_beep_stray_clips_v2/v2_20-21_35m13s.wav";
const RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_4: &str =
    "../../sample_audios/regressions/rthk_beep_stray_clips_v2/v2_22-23_19m48s.wav";

const RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_CASES: &[Case] = &[
    (RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_1, &[2.00525, 3.004875]),
    (RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_2, &[1.01525, 2.014875, 3.015]),
    (RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_3, &[0.01525, 1.014875, 2.015, 3.01225]),
];

const RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_CASES: &[Case] = &[
    (RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_1, &[]),
    (RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_2, &[]),
    (RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_3, &[]),
    (RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_4, &[]),
];

const RTHK_BEEP_HOURLY_LEADIN_12_TO_13: &str =
    "../../sample_audios/regressions/rthk_beep_hourly_leadins/radio1_2026-04-06_12_to_13_28m51_leadin.wav";
const RTHK_BEEP_HOURLY_LEADIN_17_TO_18: &str =
    "../../sample_audios/regressions/rthk_beep_hourly_leadins/radio1_2026-04-06_17_to_18_59m01_leadin.wav";

const RTHK_BEEP_HOURLY_LEADIN_CASES: &[Case] = &[
    (RTHK_BEEP_HOURLY_LEADIN_12_TO_13, &[1.0085, 2.0, 3.013125, 3.987875, 5.025125]),
    (RTHK_BEEP_HOURLY_LEADIN_17_TO_18, &[0.014125, 1.02625, 2.01, 3.015375, 4.017875]),
];

const RTHK_BEEP_HOURLY_OPENING_12_TO_13: &str =
    "../../sample_audios/regressions/rthk_beep_hourly_openings/radio1_2026-04-06_12_to_13_28m49_opening.wav";
const RTHK_BEEP_HOURLY_OPENING_17_TO_18: &str =
    "../../sample_audios/regressions/rthk_beep_hourly_openings/radio1_2026-04-06_17_to_18_58m58_opening.wav";

const RTHK_BEEP_HOURLY_OPENING_CASES: &[Case] = &[
    (
        RTHK_BEEP_HOURLY_OPENING_12_TO_13,
        &[1.02325, 2.0335, 3.025, 4.038125, 5.012875, 6.050125],
    ),
    (
        RTHK_BEEP_HOURLY_OPENING_17_TO_18,
        &[1.06975, 2.068875, 3.090625, 4.074375, 5.07975, 6.08225],
    ),
];

const RADIO903_BEEP_OPENING_RECOVERY: &str =
    "../../sample_audios/regressions/903_beep_openings/radio903_2026-04-17_09_to_10_12s_opening.wav";
const RADIO903_BEEP_OPENING_RECOVERY_15_TO_16: &str =
    "../../sample_audios/regressions/903_beep_openings/radio903_2026-04-17_15_to_16_opening.wav";
const RADIO903_BEEP_OPENING_NEGATIVE: &str =
    "../../sample_audios/regressions/903_beep_openings/radio903_2026-04-17_06_to_07_no_opening_beep.wav";
const RADIO881_BEEP_OPENING_RECOVERY: &str =
    "../../sample_audios/regressions/881_beep_openings/radio881_2026-04-16_10_to_11_10s_opening.wav";
const RADIO881_BEEP_OPENING_RECOVERY_DIRTY: &str =
    "../../sample_audios/regressions/881_beep_openings/radio881_2026-04-15_11_to_12_30m20s_opening.wav";

const RADIO903_BEEP_OPENING_CASES: &[Case] = &[
    (RADIO903_BEEP_OPENING_RECOVERY, &[12.163125]),
    (RADIO903_BEEP_OPENING_RECOVERY_15_TO_16, &[11.26425]),
];

const RADIO903_BEEP_FALSE_POSITIVE_CASES: &[Case] = &[(RADIO903_BEEP_OPENING_NEGATIVE, &[])];

const RADIO881_BEEP_OPENING_CASES: &[Case] = &[
    (RADIO881_BEEP_OPENING_RECOVERY, &[10.78125]),
    (RADIO881_BEEP_OPENING_RECOVERY_DIRTY, &[10.25875]),
];

const RADIO881_BEEP_FALSE_POSITIVE_CASES: &[Case] = &[(RADIO903_BEEP_OPENING_NEGATIVE, &[])];

// Tolerance is 0.02s: the .apd.toml pattern is a synthesised pure sine, so the
// cross-correlation peak can land at a phase-aligned offset up to ~1 cycle
// away from the true beep start (~1ms at 1 kHz, but accumulates across the
// clip). 20 ms keeps regression sensitivity without over-fitting to the
// specific phase of whichever WAV happened to generate the golden values.
const TIMESTAMP_TOLERANCE_SECONDS: f64 = 0.02;

fn case_name(audio_file: &str) -> String {
    Path::new(audio_file).file_stem().unwrap().to_string_lossy().into_owned()
}

/// Run `pattern` against `audio_file` and return the timestamps reported for `clip_name`.
fn detect(case: &str, audio_file: &str, pattern: &str, clip_name: &str) -> Vec<f64> {
    assert!(Path::new(pattern).exists(), "[{case}] Pattern file {pattern} not found");
    assert!(Path::new(audio_file).exists(), "[{case}] Audio file {audio_file} not found");

    let (peak_times, _) = match_pattern(audio_file, &[pattern], &MatchOptions::default(), None, true)
        .unwrap_or_else(|e| panic!("[{case}] match_pattern failed: {e}"));

    let peak_times = peak_times.unwrap_or_else(|| panic!("[{case}] peak_times is None"));
    peak_times
        .get(clip_name)
        .unwrap_or_else(|| panic!("[{case}] '{clip_name}' missing from peak_times: {peak_times:?}"))
        .clone()
}

fn assert_expected_timestamps(case: &str, actual_timestamps: &[f64], expected_timestamps: &[f64]) {
    assert_eq!(
        actual_timestamps.len(),
        expected_timestamps.len(),
        "[{case}] Expected {} matches, found {}: {actual_timestamps:?}",
        expected_timestamps.len(),
        actual_timestamps.len(),
    );
    let mut actual = actual_timestamps.to_vec();
    let mut expected = expected_timestamps.to_vec();
    actual.sort_by(f64::total_cmp);
    expected.sort_by(f64::total_cmp);
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!(
            (actual - expected).abs() < TIMESTAMP_TOLERANCE_SECONDS,
            "[{case}] Expected timestamp ~{expected}s, got {actual}s (all: {actual_timestamps:?})"
        );
    }
}

/// Every case must recover its expected timestamps within the tolerance.
fn check_recovered(cases: &[Case], pattern: &str, clip_name: &str) {
    for &(audio_file, expected_timestamps) in cases {
        let case = case_name(audio_file);
        let actual = detect(&case, audio_file, pattern, clip_name);
        assert_expected_timestamps(&case, &actual, expected_timestamps);
    }
}

/// Every case must report exactly its expected timestamps (an empty list for false positives).
fn check_exact(cases: &[Case], pattern: &str, clip_name: &str) {
    for &(audio_file, expected_timestamps) in cases {
        let case = case_name(audio_file);
        let actual = detect(&case, audio_file, pattern, clip_name);
        assert_eq!(actual, expected_timestamps, "[{case}] unexpected timestamps");
    }
}

#[test]
fn test_rthk_beep_stray_clips_v2_true_positives() {
    check_recovered(RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_CASES, RTHK_BEEP_PATTERN, RTHK_BEEP_CLIP_NAME);
}

#[test]
fn test_rthk_beep_stray_clips_v2_false_positives() {
    check_exact(RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_CASES, RTHK_BEEP_PATTERN, RTHK_BEEP_CLIP_NAME);
}

#[test]
fn test_rthk_beep_hourly_leadins_recover_opening_beeps() {
    check_recovered(RTHK_BEEP_HOURLY_LEADIN_CASES, RTHK_BEEP_PATTERN, RTHK_BEEP_CLIP_NAME);
}

#[test]
fn test_rthk_beep_hourly_openings_recover_first_cluster_beeps() {
    check_recovered(RTHK_BEEP_HOURLY_OPENING_CASES, RTHK_BEEP_PATTERN, RTHK_BEEP_CLIP_NAME);
}

#[test]
fn test_radio903_marker_tone_recover_opening_beep() {
    check_recovered(RADIO903_BEEP_OPENING_CASES, RADIO903_BEEP_PATTERN, RADIO903_BEEP_CLIP_NAME);
}

#[test]
fn test_radio903_marker_tone_avoids_false_positive_openings() {
    check_exact(RADIO903_BEEP_FALSE_POSITIVE_CASES, RADIO903_BEEP_PATTERN, RADIO903_BEEP_CLIP_NAME);
}

#[test]
fn test_radio881_marker_tone_recover_opening_beep() {
    check_recovered(RADIO881_BEEP_OPENING_CASES, RADIO881_BEEP_PATTERN, RADIO881_BEEP_CLIP_NAME);
}

#[test]
fn test_radio881_marker_tone_avoids_false_positive_openings() {
    check_exact(RADIO881_BEEP_FALSE_POSITIVE_CASES, RADIO881_BEEP_PATTERN, RADIO881_BEEP_CLIP_NAME);
}
