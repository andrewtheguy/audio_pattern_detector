//! Port of `tests/test_integration_matching.py`: `match_pattern` on the sample
//! audio, 16 kHz handling, streaming through `AudioPatternDetector`, and
//! ffmpeg-free WAV file processing.

use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use audio_pattern_detector::ffmpeg::FfmpegSource;
use audio_pattern_detector::stream::{resample_audio, WavFileSource};
use audio_pattern_detector::wav::{load_wav_file, write_wav_file, WavReader, WavSpec};
use audio_pattern_detector::{
    match_pattern, AudioClip, AudioPatternDetector, AudioStream, DetectorOptions, MatchOptions, PeakTimes,
    SampleSource, DEFAULT_TARGET_SAMPLE_RATE,
};

// --- Test Data Constants ---
// Centralised paths so swapping a clip only requires editing one place.

const CBS_NEWS_PATTERN: &str = "sample_audios/clips/cbs_news.wav";
const CBS_NEWS_AUDIO: &str = "sample_audios/cbs_news_audio_section.wav";
const CBS_NEWS_NAME: &str = "cbs_news";
const CBS_NEWS_EXPECTED_TIME: f64 = 25.89875;

const RTHK_BEEP_PATTERN: &str = "sample_audios/clips/rthk_beep.apd.toml";
const RTHK_BEEP_AUDIO: &str = "sample_audios/rthk_section_with_beep.wav";
const RTHK_BEEP_NAME: &str = "rthk_beep";
const RTHK_BEEP_EXPECTED_TIMES: [f64; 2] = [1.407375, 2.419125];

const RAINBOW_INTRO_PATTERN: &str = "sample_audios/clips/天空下的彩虹intro.wav";
const RAINBOW_INTRO_AUDIO: &str = "sample_audios/am1430_section_with_rainbow_intro.wav";
const RAINBOW_INTRO_NAME: &str = "天空下的彩虹intro";
const RAINBOW_INTRO_EXPECTED_TIME: f64 = 13.848;

const RTHK_BEEP_AUDIO_16K: &str = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav";
const CBS_NEWS_AUDIO_16K: &str = "sample_audios/test_16khz/cbs_news_audio_section_16k.wav";
const CBS_NEWS_PATTERN_16K: &str = "sample_audios/test_16khz/clips/cbs_news_16k.wav";

const NONEXISTENT_PATTERN: &str = "sample_audios/clips/nonexistent.wav";
const NONEXISTENT_AUDIO: &str = "sample_audios/nonexistent.wav";
const NONEXISTENT_WAV: &str = "nonexistent.wav";

const SR: u32 = DEFAULT_TARGET_SAMPLE_RATE;

// --- Helpers ---

fn assert_exists(path: &str) {
    assert!(Path::new(path).exists(), "File {path} not found");
}

/// `match_pattern(audio, patterns, debug_mode=False)` with accumulated results.
fn run_match(audio_file: &str, pattern_files: &[&str]) -> (PeakTimes, f64) {
    let (peak_times, total_time) =
        match_pattern(audio_file, pattern_files, &MatchOptions::default(), None, true).unwrap();
    (peak_times.expect("results are accumulated"), total_time)
}

fn match_error(audio_file: &str, pattern_files: &[&str]) -> String {
    match_pattern(audio_file, pattern_files, &MatchOptions::default(), None, true)
        .expect_err("match_pattern should fail")
        .to_string()
}

fn matches<'a>(peak_times: &'a PeakTimes, name: &str) -> &'a [f64] {
    peak_times
        .get(name)
        .unwrap_or_else(|| panic!("{name} key should exist in results: {peak_times:?}"))
}

fn sorted(times: &[f64]) -> Vec<f64> {
    let mut times = times.to_vec();
    times.sort_by(f64::total_cmp);
    times
}

/// Assert `actual` (sorted) has exactly the `expected` timestamps, each within `tolerance`.
fn assert_times(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "Expected {} matches, found {actual:?}", expected.len());
    for (i, (actual, expected)) in sorted(actual).iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < tolerance,
            "Match {i}: Expected timestamp ~{expected}s, got {actual}s"
        );
    }
}

fn stem(path: &str) -> String {
    Path::new(path).file_stem().unwrap().to_string_lossy().into_owned()
}

fn load_clip(pattern_file: &str) -> AudioClip {
    AudioClip::from_audio_file(pattern_file, SR).unwrap()
}

/// Decode `audio_file` through ffmpeg to mono float32 at 8 kHz and run the
/// detector on that stream (Python: `ffmpeg_get_float32_pcm` + `AudioStream`).
fn stream_detect(audio_file: &str, pattern_files: &[&str], seconds_per_chunk: Option<u32>) -> (PeakTimes, f64) {
    let clips: Vec<AudioClip> = pattern_files.iter().map(|pf| load_clip(pf)).collect();
    let mut options = DetectorOptions::default();
    if seconds_per_chunk.is_some() {
        options.seconds_per_chunk = seconds_per_chunk;
    }
    let detector = AudioPatternDetector::new(clips, options).unwrap();

    let mut source = FfmpegSource::open(audio_file, SR).unwrap();
    let (peak_times, total_time) = {
        let mut stream = AudioStream::new(stem(audio_file), &mut source, SR);
        detector.find_clip_in_audio(&mut stream, None, true).unwrap()
    };
    source.finish().unwrap();
    (peak_times.expect("results are accumulated"), total_time)
}

fn wav_spec(path: &str) -> WavSpec {
    WavReader::new(BufReader::new(File::open(path).unwrap())).unwrap().spec()
}

/// Read a source to EOF in reads of `samples_per_read` samples.
fn read_all(source: &mut WavFileSource, samples_per_read: usize) -> Vec<f32> {
    let mut all = Vec::new();
    loop {
        let chunk = source.read_samples(samples_per_read).unwrap();
        if chunk.is_empty() {
            return all;
        }
        all.extend(chunk);
    }
}

// --- Pattern Matching Tests ---

// Marker tone (pure tone beep) detection.
#[test]
fn test_rthk_beep_pattern_detection() {
    assert_exists(RTHK_BEEP_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO);

    let (peak_times, total_time) = run_match(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN]);

    assert_times(matches(&peak_times, RTHK_BEEP_NAME), &RTHK_BEEP_EXPECTED_TIMES, 0.01);

    assert!(total_time > 0.0, "Total processing time should be positive");
    assert!(total_time < 10.0, "Processing took too long: {total_time}s");
}

// Normal pattern detection.
#[test]
fn test_cbs_news_pattern_detection() {
    assert_exists(CBS_NEWS_PATTERN);
    assert_exists(CBS_NEWS_AUDIO);

    let (peak_times, total_time) = run_match(CBS_NEWS_AUDIO, &[CBS_NEWS_PATTERN]);

    assert_times(matches(&peak_times, CBS_NEWS_NAME), &[CBS_NEWS_EXPECTED_TIME], 0.01);
    assert!(total_time > 0.0, "Total processing time should be positive");
}

#[test]
fn test_multiple_patterns_detection() {
    let (peak_times_cbs, _) = run_match(CBS_NEWS_AUDIO, &[CBS_NEWS_PATTERN]);
    assert_times(matches(&peak_times_cbs, CBS_NEWS_NAME), &[CBS_NEWS_EXPECTED_TIME], 0.01);

    let (peak_times_rainbow, _) = run_match(RAINBOW_INTRO_AUDIO, &[RAINBOW_INTRO_PATTERN]);
    assert_times(matches(&peak_times_rainbow, RAINBOW_INTRO_NAME), &[RAINBOW_INTRO_EXPECTED_TIME], 1.0);
}

#[test]
fn test_pattern_not_in_audio() {
    assert_exists(CBS_NEWS_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO);

    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &[CBS_NEWS_PATTERN]);

    assert_eq!(matches(&peak_times, CBS_NEWS_NAME), &[] as &[f64]);
}

#[test]
fn test_nonexistent_pattern_file() {
    let err = match_error(RTHK_BEEP_AUDIO, &[NONEXISTENT_PATTERN]);
    assert!(err.contains("does not exist"), "unexpected error: {err}");
}

#[test]
fn test_nonexistent_audio_file() {
    let err = match_error(NONEXISTENT_AUDIO, &[RTHK_BEEP_PATTERN]);
    assert!(err.contains("does not exist"), "unexpected error: {err}");
}

#[test]
fn test_empty_pattern_list() {
    let err = match_error(RTHK_BEEP_AUDIO, &[]);
    assert!(err.contains("No pattern clips passed"), "unexpected error: {err}");
}

#[test]
fn test_beep_detection_algorithm_specifics() {
    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN]);
    let matches = matches(&peak_times, RTHK_BEEP_NAME);

    assert_eq!(matches.len(), 2, "Beep algorithm should find exactly 2 matches");
    assert!(matches[0] < matches[1], "Matches should be sorted chronologically");

    let time_diff = matches[1] - matches[0];
    assert!(0.5 < time_diff && time_diff < 5.0, "Beeps should be 0.5-5s apart, got {time_diff}s");
}

#[test]
fn test_normal_pattern_detection_algorithm_specifics() {
    let (peak_times, _) = run_match(CBS_NEWS_AUDIO, &[CBS_NEWS_PATTERN]);
    let matches = matches(&peak_times, CBS_NEWS_NAME);

    assert_eq!(matches.len(), 1, "Normal pattern algorithm should find exactly 1 match");
    assert!(matches[0] > 20.0, "CBS news pattern should be found after 20s, got {}s", matches[0]);
}

#[test]
fn test_correlation_peak_finding() {
    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN]);
    let matches = matches(&peak_times, RTHK_BEEP_NAME);

    assert!(!matches.is_empty(), "Should find at least one peak");
    for m in matches {
        assert!(*m >= 0.0, "Timestamp should be non-negative, got {m}");
    }
}

#[test]
fn test_loudness_normalization_effect() {
    let (peak_times, _) = run_match(CBS_NEWS_AUDIO, &[CBS_NEWS_PATTERN]);

    assert!(
        !matches(&peak_times, CBS_NEWS_NAME).is_empty(),
        "Loudness normalization should not prevent pattern detection"
    );
}

// --- No Matching Patterns Tests ---

#[test]
fn test_beep_pattern_in_normal_audio() {
    assert_exists(RTHK_BEEP_PATTERN);
    assert_exists(CBS_NEWS_AUDIO);

    let (peak_times, _) = run_match(CBS_NEWS_AUDIO, &[RTHK_BEEP_PATTERN]);

    assert_eq!(matches(&peak_times, RTHK_BEEP_NAME), &[] as &[f64], "RTHK beep should not match CBS news audio");
}

#[test]
fn test_cbs_pattern_in_rthk_audio() {
    assert_exists(CBS_NEWS_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO);

    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &[CBS_NEWS_PATTERN]);

    assert_eq!(matches(&peak_times, CBS_NEWS_NAME), &[] as &[f64], "CBS news should not match RTHK audio");
}

#[test]
fn test_multiple_patterns_none_match() {
    let pattern_files = [CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN];
    for pattern_file in pattern_files {
        assert_exists(pattern_file);
    }
    assert_exists(RTHK_BEEP_AUDIO);

    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &pattern_files);

    assert_eq!(matches(&peak_times, CBS_NEWS_NAME), &[] as &[f64], "CBS news should not match");
    assert_eq!(matches(&peak_times, RAINBOW_INTRO_NAME), &[] as &[f64], "Rainbow intro should not match");
}

// Each pattern only matches its own audio and produces no false positives on the others.
#[test]
fn test_all_available_patterns_mixed_results() {
    let all_patterns = [RTHK_BEEP_PATTERN, CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN];

    assert_exists(RTHK_BEEP_AUDIO);
    let (rthk_results, _) = run_match(RTHK_BEEP_AUDIO, &all_patterns);
    assert_eq!(matches(&rthk_results, RTHK_BEEP_NAME).len(), 2, "RTHK beep should match in RTHK audio");
    assert_eq!(matches(&rthk_results, CBS_NEWS_NAME).len(), 0, "CBS news should not match in RTHK audio");
    assert_eq!(matches(&rthk_results, RAINBOW_INTRO_NAME).len(), 0, "Rainbow intro should not match in RTHK audio");

    assert_exists(CBS_NEWS_AUDIO);
    let (cbs_results, _) = run_match(CBS_NEWS_AUDIO, &all_patterns);
    assert_eq!(matches(&cbs_results, CBS_NEWS_NAME).len(), 1, "CBS news should match in CBS audio");
    assert_eq!(matches(&cbs_results, RTHK_BEEP_NAME).len(), 0, "RTHK beep should not match in CBS audio");
    assert_eq!(matches(&cbs_results, RAINBOW_INTRO_NAME).len(), 0, "Rainbow intro should not match in CBS audio");

    assert_exists(RAINBOW_INTRO_AUDIO);
    let (am_results, _) = run_match(RAINBOW_INTRO_AUDIO, &all_patterns);
    assert_eq!(matches(&am_results, RAINBOW_INTRO_NAME).len(), 1, "Rainbow intro should match in AM1430 audio");
    assert_eq!(matches(&am_results, CBS_NEWS_NAME).len(), 0, "CBS news should not match in AM1430 audio");
    assert_eq!(matches(&am_results, RTHK_BEEP_NAME).len(), 0, "RTHK beep should not match in AM1430 audio");
}

#[test]
fn test_similarity_threshold_rejection() {
    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &[CBS_NEWS_PATTERN]);

    assert_eq!(
        matches(&peak_times, CBS_NEWS_NAME),
        &[] as &[f64],
        "Pattern should be rejected due to similarity threshold"
    );
}

#[test]
fn test_beep_rejection_in_non_matching_audio() {
    let (peak_times, _) = run_match(CBS_NEWS_AUDIO, &[RTHK_BEEP_PATTERN]);

    assert_eq!(
        matches(&peak_times, RTHK_BEEP_NAME),
        &[] as &[f64],
        "Beep pattern should not match in CBS news audio"
    );
}

#[test]
fn test_no_false_positives_in_complex_scenario() {
    let test_cases = [
        (RTHK_BEEP_PATTERN, CBS_NEWS_AUDIO, RTHK_BEEP_NAME),
        (CBS_NEWS_PATTERN, RTHK_BEEP_AUDIO, CBS_NEWS_NAME),
        (RAINBOW_INTRO_PATTERN, CBS_NEWS_AUDIO, RAINBOW_INTRO_NAME),
        (RAINBOW_INTRO_PATTERN, RTHK_BEEP_AUDIO, RAINBOW_INTRO_NAME),
    ];

    for (pattern_file, audio_file, pattern_name) in test_cases {
        assert_exists(pattern_file);
        assert_exists(audio_file);

        let (peak_times, _) = run_match(audio_file, &[pattern_file]);

        assert_eq!(
            matches(&peak_times, pattern_name),
            &[] as &[f64],
            "False positive detected: {pattern_name} in {audio_file}"
        );
    }
}

#[test]
fn test_correlation_peak_height_threshold() {
    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &[CBS_NEWS_PATTERN]);

    assert_eq!(
        matches(&peak_times, CBS_NEWS_NAME),
        &[] as &[f64],
        "Low correlation peaks should not produce matches"
    );
}

#[test]
fn test_verification_stage_filters_false_positives() {
    let (peak_times, _) = run_match(CBS_NEWS_AUDIO, &[RTHK_BEEP_PATTERN]);

    assert_eq!(
        matches(&peak_times, RTHK_BEEP_NAME),
        &[] as &[f64],
        "Verification stage should filter out false positives"
    );
}

// --- AM1430 Rainbow Intro (Lossy-Encoded Audio) Tests ---

// The audio went through Opus encoding, which degrades the cross-correlation shape.
#[test]
fn test_rainbow_intro_pattern_detection() {
    assert_exists(RAINBOW_INTRO_PATTERN);
    assert_exists(RAINBOW_INTRO_AUDIO);

    let (peak_times, total_time) = run_match(RAINBOW_INTRO_AUDIO, &[RAINBOW_INTRO_PATTERN]);

    assert_times(matches(&peak_times, RAINBOW_INTRO_NAME), &[RAINBOW_INTRO_EXPECTED_TIME], 1.0);
    assert!(total_time > 0.0, "Total processing time should be positive");
}

#[test]
fn test_rainbow_intro_not_in_rthk_audio() {
    assert_exists(RAINBOW_INTRO_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO);

    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &[RAINBOW_INTRO_PATTERN]);

    assert_eq!(matches(&peak_times, RAINBOW_INTRO_NAME), &[] as &[f64]);
}

#[test]
fn test_rainbow_intro_not_in_cbs_audio() {
    assert_exists(RAINBOW_INTRO_PATTERN);
    assert_exists(CBS_NEWS_AUDIO);

    let (peak_times, _) = run_match(CBS_NEWS_AUDIO, &[RAINBOW_INTRO_PATTERN]);

    assert_eq!(matches(&peak_times, RAINBOW_INTRO_NAME), &[] as &[f64]);
}

#[test]
fn test_cbs_pattern_not_in_rainbow_intro_audio() {
    let test_cases = [(CBS_NEWS_PATTERN, CBS_NEWS_NAME), (RTHK_BEEP_PATTERN, RTHK_BEEP_NAME)];
    assert_exists(RAINBOW_INTRO_AUDIO);

    for (pattern_file, pattern_name) in test_cases {
        assert_exists(pattern_file);

        let (peak_times, _) = run_match(RAINBOW_INTRO_AUDIO, &[pattern_file]);

        assert_eq!(
            matches(&peak_times, pattern_name),
            &[] as &[f64],
            "False positive: {pattern_name} in AM1430 audio"
        );
    }
}

// --- 16kHz Audio Handling Tests ---

#[test]
fn test_match_16khz_audio_with_8khz_pattern() {
    assert_exists(RTHK_BEEP_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO_16K);

    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO_16K, &[RTHK_BEEP_PATTERN]);

    // Tolerance increased for resampling.
    assert_times(matches(&peak_times, RTHK_BEEP_NAME), &RTHK_BEEP_EXPECTED_TIMES, 0.05);
}

#[test]
fn test_match_16khz_cbs_news() {
    assert_exists(CBS_NEWS_PATTERN);
    assert_exists(CBS_NEWS_AUDIO_16K);

    let (peak_times, _) = run_match(CBS_NEWS_AUDIO_16K, &[CBS_NEWS_PATTERN]);

    assert_times(matches(&peak_times, CBS_NEWS_NAME), &[CBS_NEWS_EXPECTED_TIME], 0.05);
}

// `.apd.toml` patterns synthesise the clip at the target sample rate, so the
// same pattern file works for 8kHz and 16kHz audio without any pre-conversion.
#[test]
fn test_match_16khz_with_apd_pattern() {
    assert_exists(RTHK_BEEP_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO_16K);

    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO_16K, &[RTHK_BEEP_PATTERN]);

    assert_eq!(matches(&peak_times, RTHK_BEEP_NAME).len(), 2);
}

// A `.wav` pattern (cbs_news, converted to 8kHz) that matches together with an
// `.apd.toml` pure-tone pattern (rthk_beep) that should not match CBS news audio.
#[test]
fn test_multiple_patterns_mixed_formats() {
    assert_exists(CBS_NEWS_PATTERN_16K);
    assert_exists(RTHK_BEEP_PATTERN);

    let converted = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
    let (audio, source_sr) = load_wav_file(CBS_NEWS_PATTERN_16K).unwrap();
    let audio = resample_audio(audio, source_sr, 8000);
    write_wav_file(converted.path(), &audio, 8000).unwrap();
    let converted_path = converted.path().to_str().unwrap();

    assert_exists(CBS_NEWS_AUDIO_16K);
    let (peak_times, _) = run_match(CBS_NEWS_AUDIO_16K, &[converted_path, RTHK_BEEP_PATTERN]);

    assert_eq!(peak_times.len(), 2, "Expected 2 pattern results");
    let mut pattern_match_counts: Vec<usize> = peak_times.values().map(Vec::len).collect();
    pattern_match_counts.sort_unstable();
    assert_eq!(
        pattern_match_counts,
        vec![0, 1],
        "Expected one pattern with 1 match and one with 0, got {peak_times:?}"
    );
}

#[test]
fn test_16khz_no_false_positives() {
    assert_exists(CBS_NEWS_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO_16K);

    let (peak_times, _) = run_match(RTHK_BEEP_AUDIO_16K, &[CBS_NEWS_PATTERN]);

    assert_eq!(
        matches(&peak_times, CBS_NEWS_NAME),
        &[] as &[f64],
        "16kHz conversion should not introduce false positives"
    );
}

#[test]
fn test_16khz_beep_pattern_rejection() {
    assert_exists(RTHK_BEEP_PATTERN);
    assert_exists(CBS_NEWS_AUDIO_16K);

    let (peak_times, _) = run_match(CBS_NEWS_AUDIO_16K, &[RTHK_BEEP_PATTERN]);

    assert_eq!(
        matches(&peak_times, RTHK_BEEP_NAME),
        &[] as &[f64],
        "Beep algorithm should reject mismatches in 16kHz audio"
    );
}

// Timestamps reflect the original audio timeline, not the converted one.
#[test]
fn test_sample_rate_preservation_in_results() {
    assert_exists(RTHK_BEEP_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO);
    assert_exists(RTHK_BEEP_AUDIO_16K);

    let (results_8k, _) = run_match(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN]);
    let (results_16k, _) = run_match(RTHK_BEEP_AUDIO_16K, &[RTHK_BEEP_PATTERN]);

    let times_8k = sorted(matches(&results_8k, RTHK_BEEP_NAME));
    let times_16k = sorted(matches(&results_16k, RTHK_BEEP_NAME));
    assert_eq!(times_8k.len(), times_16k.len(), "Different sample rates should find same number of matches");

    for (i, (time_8k, time_16k)) in times_8k.iter().zip(&times_16k).enumerate() {
        assert!(
            (time_8k - time_16k).abs() < 0.1,
            "Match {i}: Timestamps differ too much: 8kHz={time_8k}s, 16kHz={time_16k}s"
        );
    }
}

// --- Streaming Audio Processing Tests ---

#[test]
fn test_streaming_rthk_beep_detection() {
    assert_exists(RTHK_BEEP_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO);

    let (peak_times, _) = stream_detect(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], None);

    assert_times(matches(&peak_times, RTHK_BEEP_NAME), &RTHK_BEEP_EXPECTED_TIMES, 0.01);
}

#[test]
fn test_streaming_cbs_news_detection() {
    assert_exists(CBS_NEWS_PATTERN);
    assert_exists(CBS_NEWS_AUDIO);

    let (peak_times, _) = stream_detect(CBS_NEWS_AUDIO, &[CBS_NEWS_PATTERN], None);

    assert_times(matches(&peak_times, CBS_NEWS_NAME), &[CBS_NEWS_EXPECTED_TIME], 0.01);
}

#[test]
fn test_streaming_multiple_patterns() {
    let pattern_files = [CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN];
    assert_exists(CBS_NEWS_AUDIO);
    for pf in pattern_files {
        assert_exists(pf);
    }

    let (peak_times, _) = stream_detect(CBS_NEWS_AUDIO, &pattern_files, None);

    assert_eq!(matches(&peak_times, CBS_NEWS_NAME).len(), 1);
    assert_eq!(matches(&peak_times, RAINBOW_INTRO_NAME).len(), 0);
}

// ffmpeg converts 16kHz to 8kHz while streaming.
#[test]
fn test_streaming_16khz_audio_conversion() {
    assert_exists(RTHK_BEEP_PATTERN);
    assert_exists(RTHK_BEEP_AUDIO_16K);

    let (peak_times, _) = stream_detect(RTHK_BEEP_AUDIO_16K, &[RTHK_BEEP_PATTERN], None);

    assert_times(matches(&peak_times, RTHK_BEEP_NAME), &RTHK_BEEP_EXPECTED_TIMES, 0.05);
}

#[test]
fn test_streaming_chunk_processing() {
    let (peak_times, _) = stream_detect(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], Some(60));

    assert_eq!(matches(&peak_times, RTHK_BEEP_NAME).len(), 2);
}

// With small chunks and sliding window overlap the same pattern may be detected
// in multiple chunks, so duplicates may exist; the expected timestamps must be found.
#[test]
fn test_streaming_small_chunk_size() {
    // rthk_beep is ~0.23s, so sliding_window rounds to 1s and the minimum chunk size is 2s.
    let (peak_times, _) = stream_detect(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], Some(3));
    let matches = matches(&peak_times, RTHK_BEEP_NAME);

    assert!(matches.len() >= 2);

    let found_times: Vec<f64> = RTHK_BEEP_EXPECTED_TIMES
        .iter()
        .copied()
        .filter(|expected| matches.iter().any(|actual| (actual - expected).abs() < 0.01))
        .collect();
    assert_eq!(
        found_times.len(),
        RTHK_BEEP_EXPECTED_TIMES.len(),
        "Expected to find timestamps near {RTHK_BEEP_EXPECTED_TIMES:?}, found {matches:?}"
    );
}

#[test]
fn test_streaming_no_match_scenario() {
    let (peak_times, _) = stream_detect(RTHK_BEEP_AUDIO, &[CBS_NEWS_PATTERN], None);

    assert_eq!(matches(&peak_times, CBS_NEWS_NAME), &[] as &[f64]);
}

#[test]
fn test_streaming_total_time_accuracy() {
    let (_, total_time) = stream_detect(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], None);

    // rthk_section_with_beep.wav is ~4.08 seconds
    assert!(4.0 < total_time && total_time < 4.2, "Expected ~4.08s, got {total_time}s");
}

#[test]
fn test_audio_clip_from_file() {
    let clip = load_clip(RTHK_BEEP_PATTERN);

    assert_eq!(clip.name, RTHK_BEEP_NAME);
    assert_eq!(clip.sample_rate, DEFAULT_TARGET_SAMPLE_RATE);
    assert!(!clip.audio.is_empty());
    assert!(clip.clip_length_seconds() > 0.0);
}

#[test]
fn test_audio_clip_sample_rate_validation() {
    let pattern_clip = load_clip(RTHK_BEEP_PATTERN);
    assert_eq!(pattern_clip.sample_rate, DEFAULT_TARGET_SAMPLE_RATE);

    // Detector should accept valid clips
    let detector = AudioPatternDetector::new(vec![pattern_clip], DetectorOptions::default()).unwrap();
    assert_eq!(detector.get_config().clips.len(), 1);
}

// Each pattern's results are stored under the correct key.
#[test]
fn test_streaming_maintains_pattern_order() {
    let pattern_files = [RTHK_BEEP_PATTERN, CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN];

    let (peak_times, _) = stream_detect(CBS_NEWS_AUDIO, &pattern_files, None);

    assert_eq!(matches(&peak_times, RTHK_BEEP_NAME).len(), 0);
    assert_eq!(matches(&peak_times, CBS_NEWS_NAME).len(), 1);
    assert_eq!(matches(&peak_times, RAINBOW_INTRO_NAME).len(), 0);
}

#[test]
fn test_streaming_duplicate_pattern_names_rejected() {
    let clip1 = load_clip(RTHK_BEEP_PATTERN);
    let clip2 = load_clip(RTHK_BEEP_PATTERN); // Same name

    let err = AudioPatternDetector::new(vec![clip1, clip2], DetectorOptions::default())
        .err()
        .expect("duplicate clip names should be rejected")
        .to_string();
    assert!(err.contains("needs to be unique"), "unexpected error: {err}");
}

#[test]
fn test_streaming_16khz_cbs_news() {
    assert_exists(CBS_NEWS_PATTERN);
    assert_exists(CBS_NEWS_AUDIO_16K);

    let (peak_times, _) = stream_detect(CBS_NEWS_AUDIO_16K, &[CBS_NEWS_PATTERN], None);

    assert_times(matches(&peak_times, CBS_NEWS_NAME), &[CBS_NEWS_EXPECTED_TIME], 0.05);
}

#[test]
fn test_streaming_results_match_high_level_api() {
    let (high_level_results, _) = run_match(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN]);
    let (streaming_results, _) = stream_detect(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], None);

    let high_level = sorted(matches(&high_level_results, RTHK_BEEP_NAME));
    let streaming = sorted(matches(&streaming_results, RTHK_BEEP_NAME));
    assert_eq!(high_level.len(), streaming.len());

    for (hl, st) in high_level.iter().zip(&streaming) {
        assert!((hl - st).abs() < 0.001, "Results differ: high-level={hl}, streaming={st}");
    }
}

// --- WAV File Processing Without FFmpeg Tests ---

/// Tests for `WavFileSource` (Python: `_WavFileStreamWrapper`, ffmpeg-free WAV file streaming).
mod test_wav_file_stream_wrapper {
    use super::*;

    #[test]
    fn test_wav_file_stream_wrapper_basic() {
        assert_exists(CBS_NEWS_PATTERN);

        let source = WavFileSource::open(CBS_NEWS_PATTERN, DEFAULT_TARGET_SAMPLE_RATE).unwrap();
        assert!(source.input_sample_rate() > 0);

        // WavFileSource does not expose channels / sample width; read them from the header.
        let spec = wav_spec(CBS_NEWS_PATTERN);
        assert!(spec.channels >= 1);
        assert!(spec.format.bytes_per_sample() > 0);
    }

    #[test]
    fn test_wav_file_stream_wrapper_read() {
        assert_exists(CBS_NEWS_PATTERN);

        let mut source = WavFileSource::open(CBS_NEWS_PATTERN, DEFAULT_TARGET_SAMPLE_RATE).unwrap();
        let audio = source.read_samples(1000).unwrap();
        assert!(!audio.is_empty());

        // Audio should be normalized
        let max_abs = audio.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        assert!(max_abs <= 1.5);
    }

    #[test]
    fn test_wav_file_stream_wrapper_full_read() {
        assert_exists(CBS_NEWS_PATTERN);

        let mut source = WavFileSource::open(CBS_NEWS_PATTERN, DEFAULT_TARGET_SAMPLE_RATE).unwrap();
        let audio = read_all(&mut source, 8000);
        assert!(!audio.is_empty());
    }

    #[test]
    fn test_wav_file_stream_wrapper_resampling() {
        // Use 16kHz file to test resampling to 8kHz
        assert_exists(CBS_NEWS_PATTERN_16K);

        let mut source = WavFileSource::open(CBS_NEWS_PATTERN_16K, 8000).unwrap();
        assert_eq!(source.input_sample_rate(), 16000);
        assert!(source.needs_resample());

        let audio = read_all(&mut source, 8000);
        assert!(!audio.is_empty());
    }

    #[test]
    fn test_wav_file_stream_wrapper_no_resampling() {
        assert_exists(CBS_NEWS_PATTERN); // .wav, native 8kHz

        let source = WavFileSource::open(CBS_NEWS_PATTERN, 8000).unwrap();
        assert_eq!(source.input_sample_rate(), 8000);
        assert!(!source.needs_resample());
    }

    #[test]
    fn test_wav_file_stream_wrapper_nonexistent_file() {
        let err = WavFileSource::open(NONEXISTENT_WAV, 8000)
            .err()
            .expect("opening a nonexistent file should fail")
            .to_string();
        assert!(err.contains("Failed to read WAV file"), "unexpected error: {err}");
    }

    #[test]
    fn test_wav_file_stream_wrapper_with_audio_stream() {
        assert_exists(CBS_NEWS_PATTERN);

        let source = WavFileSource::open(CBS_NEWS_PATTERN, DEFAULT_TARGET_SAMPLE_RATE).unwrap();
        let audio_stream = AudioStream::new("test_stream", source, DEFAULT_TARGET_SAMPLE_RATE);
        assert_eq!(audio_stream.name, "test_stream");
        assert_eq!(audio_stream.sample_rate, DEFAULT_TARGET_SAMPLE_RATE);
    }
}

/// Tests for WAV file pattern matching without ffmpeg.
mod test_wav_file_matching_without_ffmpeg {
    use super::*;

    #[test]
    fn test_wav_match_without_ffmpeg() {
        assert_exists(RTHK_BEEP_PATTERN);
        assert_exists(RTHK_BEEP_AUDIO);

        // Uses WavFileSource internally
        let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN]);

        assert_times(matches(&peak_times, RTHK_BEEP_NAME), &RTHK_BEEP_EXPECTED_TIMES, 0.01);
    }

    #[test]
    fn test_wav_match_results_consistent() {
        assert_exists(CBS_NEWS_PATTERN);
        assert_exists(CBS_NEWS_AUDIO);

        let (peak_times, _) = run_match(CBS_NEWS_AUDIO, &[CBS_NEWS_PATTERN]);

        assert_times(matches(&peak_times, CBS_NEWS_NAME), &[CBS_NEWS_EXPECTED_TIME], 0.01);
    }

    #[test]
    fn test_wav_match_16khz_resampling() {
        assert_exists(RTHK_BEEP_PATTERN);
        assert_exists(RTHK_BEEP_AUDIO_16K);

        let (peak_times, _) = run_match(RTHK_BEEP_AUDIO_16K, &[RTHK_BEEP_PATTERN]);

        assert_times(matches(&peak_times, RTHK_BEEP_NAME), &RTHK_BEEP_EXPECTED_TIMES, 0.05);
    }

    #[test]
    fn test_wav_match_no_false_positives() {
        assert_exists(CBS_NEWS_PATTERN);
        assert_exists(RTHK_BEEP_AUDIO);

        let (peak_times, _) = run_match(RTHK_BEEP_AUDIO, &[CBS_NEWS_PATTERN]);

        assert_eq!(matches(&peak_times, CBS_NEWS_NAME), &[] as &[f64]);
    }

    #[test]
    fn test_wav_match_multiple_patterns() {
        let pattern_files = [CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN];
        for pf in pattern_files {
            assert_exists(pf);
        }
        assert_exists(CBS_NEWS_AUDIO);

        let (peak_times, _) = run_match(CBS_NEWS_AUDIO, &pattern_files);

        assert_eq!(matches(&peak_times, CBS_NEWS_NAME).len(), 1);
        assert_eq!(matches(&peak_times, RAINBOW_INTRO_NAME).len(), 0);
    }

    // Python mocked ffmpeg as unavailable. Here the CLI runs in a child process
    // whose PATH is an empty directory, so ffmpeg cannot be found.
    #[test]
    fn test_wav_match_without_ffmpeg_available() {
        assert_exists(RTHK_BEEP_PATTERN);
        assert_exists(RTHK_BEEP_AUDIO);

        let empty_path = tempfile::tempdir().unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_audio-pattern-detector"))
            .args(["match", RTHK_BEEP_AUDIO, "--pattern-file", RTHK_BEEP_PATTERN])
            .env("PATH", empty_path.path())
            .output()
            .unwrap();
        assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

        let events: Vec<serde_json::Value> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let detections: Vec<&serde_json::Value> =
            events.iter().filter(|event| event["type"] == "pattern_detected").collect();
        for detection in &detections {
            assert_eq!(detection["clip_name"], RTHK_BEEP_NAME);
        }
        let times: Vec<f64> = detections
            .iter()
            .map(|detection| detection["timestamp_ms"].as_f64().unwrap() / 1000.0)
            .collect();

        assert_times(&times, &RTHK_BEEP_EXPECTED_TIMES, 0.01);
    }

    #[test]
    fn test_wav_match_streaming_with_wrapper() {
        assert_exists(RTHK_BEEP_PATTERN);
        assert_exists(RTHK_BEEP_AUDIO);

        let pattern_clip = load_clip(RTHK_BEEP_PATTERN);

        let source = WavFileSource::open(RTHK_BEEP_AUDIO, DEFAULT_TARGET_SAMPLE_RATE).unwrap();
        let mut audio_stream = AudioStream::new(stem(RTHK_BEEP_AUDIO), source, DEFAULT_TARGET_SAMPLE_RATE);

        let detector = AudioPatternDetector::new(vec![pattern_clip], DetectorOptions::default()).unwrap();
        let (peak_times, _) = detector.find_clip_in_audio(&mut audio_stream, None, true).unwrap();
        let peak_times = peak_times.unwrap();

        assert_eq!(matches(&peak_times, RTHK_BEEP_NAME).len(), 2);
    }

    #[test]
    fn test_wav_match_total_time_accuracy() {
        assert_exists(RTHK_BEEP_PATTERN);
        assert_exists(RTHK_BEEP_AUDIO);

        let (_, total_time) = run_match(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN]);

        // rthk_section_with_beep.wav is ~4.08 seconds
        assert!(4.0 < total_time && total_time < 4.2, "Expected ~4.08s, got {total_time}s");
    }

    // Stereo WAV (created with ffmpeg, as in Python) is mixed to mono.
    #[test]
    fn test_wav_match_stereo_file() {
        let sample_rate: u32 = 8000;
        let duration_seconds = 1;
        let num_samples = sample_rate * duration_seconds;

        let tone = |frequency: f64, i: u32| {
            ((2.0 * std::f64::consts::PI * frequency * i as f64 / sample_rate as f64).sin() * 32767.0) as i16
        };
        let stereo_bytes: Vec<u8> = (0..num_samples)
            .flat_map(|i| [tone(440.0, i), tone(880.0, i)])
            .flat_map(i16::to_le_bytes)
            .collect();

        let stereo_file = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        let stereo_path = stereo_file.path().to_str().unwrap();

        let mut ffmpeg = Command::new("ffmpeg")
            .args(["-y", "-f", "s16le", "-ar", &sample_rate.to_string(), "-ac", "2", "-i", "pipe:"])
            .args(["-loglevel", "error", stereo_path])
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        ffmpeg.stdin.take().unwrap().write_all(&stereo_bytes).unwrap();
        assert!(ffmpeg.wait().unwrap().success());

        assert_eq!(wav_spec(stereo_path).channels, 2);

        // Read data - should be converted to mono
        let mut source = WavFileSource::open(stereo_path, sample_rate).unwrap();
        let audio = source.read_samples(1000).unwrap();
        assert!(!audio.is_empty());
    }
}
