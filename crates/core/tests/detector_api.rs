//! Tests for the direct `AudioPatternDetector` API (not via the CLI):
//! 1. Callback-based detection (`on_pattern_detected`) - equivalent to JSONL mode
//! 2. Memory optimization mode (`accumulate_results = false`)
//! 3. Combined callback + accumulate scenarios
//! 4. `get_config()`

use std::path::Path;

use audio_pattern_detector_core::ffmpeg::FfmpegSource;
use audio_pattern_detector_core::{
    AudioClip, AudioPatternDetector, AudioStream, DetectorConfig, DetectorOptions, PatternDetectedCallback,
    PeakTimes, DEFAULT_SECONDS_PER_CHUNK, DEFAULT_TARGET_SAMPLE_RATE,
};

const RTHK_BEEP_PATTERN: &str = "../../sample_audios/clips/rthk_beep.apd.toml";
const CBS_NEWS_PATTERN: &str = "../../sample_audios/clips/cbs_news.wav";
const RAINBOW_INTRO_PATTERN: &str = "../../sample_audios/clips/天空下的彩虹intro.wav";

const RTHK_BEEP_AUDIO: &str = "../../sample_audios/rthk_section_with_beep.wav";
const CBS_NEWS_AUDIO: &str = "../../sample_audios/cbs_news_audio_section.wav";

const RTHK_BEEP_NAME: &str = "rthk_beep";
const CBS_NEWS_NAME: &str = "cbs_news";
const RAINBOW_INTRO_NAME: &str = "天空下的彩虹intro";

const RTHK_BEEP_EXPECTED_TIMES: [f64; 2] = [1.4165, 2.419125];

const SR: u32 = DEFAULT_TARGET_SAMPLE_RATE;

// --- Helper Functions ---

type Events = Vec<(String, f64)>;

fn load_clips(pattern_files: &[&str]) -> Vec<AudioClip> {
    pattern_files
        .iter()
        .map(|pf| AudioClip::from_audio_file(pf, SR).unwrap())
        .collect()
}

fn detector_for(pattern_files: &[&str], options: DetectorOptions) -> AudioPatternDetector {
    AudioPatternDetector::new(load_clips(pattern_files), options).unwrap()
}

/// Decode `audio_file` through ffmpeg (mono float32 PCM, like the Python
/// tests) and run the detector over it.
fn run_detector(
    audio_file: &str,
    pattern_files: &[&str],
    callback: Option<PatternDetectedCallback>,
    accumulate_results: bool,
) -> (Option<PeakTimes>, f64) {
    let detector = detector_for(pattern_files, DetectorOptions::default());

    let mut source = FfmpegSource::open(audio_file, SR).unwrap();
    let audio_name = Path::new(audio_file).file_stem().unwrap().to_string_lossy().into_owned();
    let result = {
        let mut audio_stream = AudioStream::new(audio_name, &mut source, SR);
        detector
            .find_clip_in_audio(&mut audio_stream, callback, accumulate_results)
            .unwrap()
    };
    source.finish().unwrap();
    result
}

/// Run the detector with a callback and return events and results.
fn run_detector_with_callback(
    audio_file: &str,
    pattern_files: &[&str],
    accumulate_results: bool,
) -> (Events, Option<PeakTimes>, f64) {
    let mut events: Events = Vec::new();
    let mut callback = |clip_name: &str, timestamp: f64| events.push((clip_name.to_string(), timestamp));
    let (peak_times, total_time) =
        run_detector(audio_file, pattern_files, Some(&mut callback), accumulate_results);
    (events, peak_times, total_time)
}

/// Run the detector without a callback.
fn run_detector_without_callback(
    audio_file: &str,
    pattern_files: &[&str],
    accumulate_results: bool,
) -> (Option<PeakTimes>, f64) {
    run_detector(audio_file, pattern_files, None, accumulate_results)
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values
}

fn get_config(pattern_files: &[&str]) -> DetectorConfig {
    detector_for(pattern_files, DetectorOptions::default()).get_config()
}

fn clip_names(config: &DetectorConfig) -> Vec<&str> {
    config.clips.iter().map(|(name, _)| name.as_str()).collect()
}

// --- Callback Tests (equivalent to JSONL CLI tests) ---

#[test]
fn test_callback_basic() {
    assert!(Path::new(RTHK_BEEP_PATTERN).exists());
    assert!(Path::new(RTHK_BEEP_AUDIO).exists());

    let (events, _peak_times, _total_time) =
        run_detector_with_callback(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], true);

    // Callback should have been called twice (2 beeps)
    assert_eq!(events.len(), 2, "Expected 2 callback events, got {}", events.len());

    for (clip_name, timestamp) in &events {
        assert_eq!(clip_name, RTHK_BEEP_NAME);
        assert!(*timestamp >= 0.0);
    }

    for (i, (_, actual)) in events.iter().enumerate() {
        let expected = RTHK_BEEP_EXPECTED_TIMES[i];
        assert!(
            (actual - expected).abs() < 0.01,
            "Event {i}: Expected ~{expected}s, got {actual}s"
        );
    }
}

#[test]
fn test_callback_timestamps_monotonic() {
    let (events, _, _) = run_detector_with_callback(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], true);

    assert!(events.len() >= 2, "Expected at least 2 events");

    let timestamps: Vec<f64> = events.iter().map(|(_, ts)| *ts).collect();
    for i in 1..timestamps.len() {
        assert!(
            timestamps[i] >= timestamps[i - 1],
            "Timestamps not monotonic: {} -> {}",
            timestamps[i - 1],
            timestamps[i]
        );
    }
}

#[test]
fn test_callback_multiple_patterns_monotonic() {
    let pattern_files = [
        RTHK_BEEP_PATTERN, // Found at ~1.4s and ~2.4s
        CBS_NEWS_PATTERN,  // Not found in RTHK audio
    ];

    let (events, _, _) = run_detector_with_callback(RTHK_BEEP_AUDIO, &pattern_files, true);

    assert_eq!(events.len(), 2, "Expected 2 events, got {}", events.len());

    // Both events should be rthk_beep (cbs_news doesn't match in this audio)
    for (clip_name, _) in &events {
        assert_eq!(clip_name, RTHK_BEEP_NAME, "Expected rthk_beep, got {clip_name}");
    }

    let first_ts = events[0].1;
    let second_ts = events[1].1;
    assert!(
        first_ts < second_ts,
        "Timestamps not monotonic: {first_ts} should be < {second_ts}"
    );
}

#[test]
fn test_callback_no_matches() {
    // CBS pattern not in RTHK audio
    let (events, peak_times, _) = run_detector_with_callback(RTHK_BEEP_AUDIO, &[CBS_NEWS_PATTERN], true);

    assert_eq!(events.len(), 0, "Expected 0 events for no match, got {}", events.len());

    // peak_times should have an empty list for the pattern
    let peak_times = peak_times.expect("peak_times should not be None");
    assert_eq!(peak_times.get(CBS_NEWS_NAME), Some(&Vec::new()));
}

#[test]
fn test_callback_multiple_patterns_non_matching_ignored() {
    let pattern_files = [
        RTHK_BEEP_PATTERN, // Not found in CBS audio
        CBS_NEWS_PATTERN,  // Found at ~25.9s
    ];

    let (events, _, _) = run_detector_with_callback(CBS_NEWS_AUDIO, &pattern_files, true);

    // Only cbs_news should match
    assert_eq!(events.len(), 1, "Expected 1 event, got {}: {events:?}", events.len());
    assert_eq!(events[0].0, CBS_NEWS_NAME, "Expected cbs_news, got {}", events[0].0);
}

// --- accumulate_results Tests ---

#[test]
fn test_accumulate_results_true() {
    let (peak_times, _total_time) = run_detector_without_callback(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], true);

    let peak_times = peak_times.expect("peak_times should not be None");
    let beep_times = peak_times.get(RTHK_BEEP_NAME).expect("rthk_beep key should exist");
    assert_eq!(beep_times.len(), 2, "Expected 2 matches, got {}", beep_times.len());

    for (i, (actual, expected)) in sorted(beep_times).iter().zip(RTHK_BEEP_EXPECTED_TIMES).enumerate() {
        assert!(
            (actual - expected).abs() < 0.01,
            "Match {i}: Expected ~{expected}s, got {actual}s"
        );
    }
}

#[test]
fn test_accumulate_results_false_returns_none() {
    let (peak_times, total_time) = run_detector_without_callback(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], false);

    assert!(peak_times.is_none(), "peak_times should be None, got {peak_times:?}");

    // total_time should still be valid
    assert!(total_time > 0.0, "total_time should be positive");
}

#[test]
fn test_accumulate_results_false_with_callback() {
    let (events, peak_times, _total_time) =
        run_detector_with_callback(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], false);

    // Callback should still capture events
    assert_eq!(events.len(), 2, "Expected 2 events, got {}", events.len());

    assert!(peak_times.is_none(), "peak_times should be None, got {peak_times:?}");

    for (i, (clip_name, timestamp)) in events.iter().enumerate() {
        assert_eq!(clip_name, RTHK_BEEP_NAME);
        assert!((timestamp - RTHK_BEEP_EXPECTED_TIMES[i]).abs() < 0.01);
    }
}

// Essentially a no-op mode: detection runs but results aren't saved.
#[test]
fn test_accumulate_results_false_no_callback() {
    let (peak_times, total_time) = run_detector_without_callback(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], false);

    assert!(peak_times.is_none(), "peak_times should be None, got {peak_times:?}");

    // total_time should still be tracked
    assert!(total_time > 0.0, "total_time should be positive");
    assert!(4.0 < total_time && total_time < 4.2, "Expected ~4.08s, got {total_time}s");
}

// --- Combined Callback + Accumulate Tests ---

#[test]
fn test_callback_with_accumulate_true() {
    let (events, peak_times, _total_time) =
        run_detector_with_callback(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], true);

    // Both should have captured results
    assert_eq!(events.len(), 2, "Callback should have 2 events, got {}", events.len());
    let peak_times = peak_times.expect("peak_times should not be None");
    let beep_times = &peak_times[RTHK_BEEP_NAME];
    assert_eq!(beep_times.len(), 2, "peak_times should have 2 matches, got {}", beep_times.len());

    // Callback and accumulation captured the same timestamps
    let callback_timestamps = sorted(&events.iter().map(|(_, ts)| *ts).collect::<Vec<_>>());
    let accumulated_timestamps = sorted(beep_times);

    assert_eq!(callback_timestamps.len(), accumulated_timestamps.len());
    for (cb_ts, acc_ts) in callback_timestamps.iter().zip(&accumulated_timestamps) {
        assert!(
            (cb_ts - acc_ts).abs() < 0.001,
            "Callback ({cb_ts}) and accumulated ({acc_ts}) timestamps differ"
        );
    }
}

// Streaming mode.
#[test]
fn test_callback_with_accumulate_false() {
    let pattern_files = [
        CBS_NEWS_PATTERN,
        RAINBOW_INTRO_PATTERN, // Does not match CBS audio
    ];

    let (events, peak_times, _total_time) = run_detector_with_callback(CBS_NEWS_AUDIO, &pattern_files, false);

    // Only cbs_news should match
    assert_eq!(events.len(), 1, "Expected 1 event, got {}", events.len());

    assert!(peak_times.is_none(), "peak_times should be None in streaming mode");

    assert_eq!(events[0].0, CBS_NEWS_NAME);
}

#[test]
fn test_callback_multiple_patterns_with_accumulate() {
    let pattern_files = [
        RTHK_BEEP_PATTERN,
        CBS_NEWS_PATTERN,
        RAINBOW_INTRO_PATTERN, // Does not match CBS audio
    ];

    let (events, peak_times, _total_time) = run_detector_with_callback(CBS_NEWS_AUDIO, &pattern_files, true);

    // Only cbs_news should match; RTHK and rainbow intro should not
    assert_eq!(events.len(), 1, "Expected 1 event (cbs_news only), got {}", events.len());

    let peak_times = peak_times.expect("peak_times should not be None");
    assert!(peak_times.contains_key(RTHK_BEEP_NAME));
    assert!(peak_times.contains_key(CBS_NEWS_NAME));
    assert!(peak_times.contains_key(RAINBOW_INTRO_NAME));

    // RTHK and rainbow intro should have no matches
    assert_eq!(peak_times[RTHK_BEEP_NAME].len(), 0);
    assert_eq!(peak_times[RAINBOW_INTRO_NAME].len(), 0);
    // CBS should have 1 match
    assert_eq!(peak_times[CBS_NEWS_NAME].len(), 1);

    // Callback captured the same
    let callback_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert!(callback_names.contains(&CBS_NEWS_NAME));
    assert!(!callback_names.contains(&RAINBOW_INTRO_NAME));
    assert!(!callback_names.contains(&RTHK_BEEP_NAME));
}

#[test]
fn test_callback_with_no_match_accumulate_true() {
    let (events, peak_times, _total_time) = run_detector_with_callback(RTHK_BEEP_AUDIO, &[CBS_NEWS_PATTERN], true);

    assert_eq!(events.len(), 0, "No callback events expected");
    let peak_times = peak_times.expect("peak_times should not be None");
    assert_eq!(peak_times.get(CBS_NEWS_NAME), Some(&Vec::new()));
}

#[test]
fn test_callback_with_no_match_accumulate_false() {
    let (events, peak_times, _total_time) = run_detector_with_callback(RTHK_BEEP_AUDIO, &[CBS_NEWS_PATTERN], false);

    assert_eq!(events.len(), 0, "No callback events expected");
    assert!(peak_times.is_none(), "peak_times should be None");
}

// --- Edge Cases ---

// The callback is called as patterns are detected, in order.
#[test]
fn test_callback_called_immediately() {
    let mut events_with_order: Vec<(usize, String, f64)> = Vec::new();
    let mut counter = 0;
    let mut callback = |clip_name: &str, timestamp: f64| {
        counter += 1;
        events_with_order.push((counter, clip_name.to_string(), timestamp));
    };

    run_detector(RTHK_BEEP_AUDIO, &[RTHK_BEEP_PATTERN], Some(&mut callback), true);

    assert_eq!(events_with_order.len(), 2);
    assert_eq!(events_with_order[0].0, 1); // First event
    assert_eq!(events_with_order[1].0, 2); // Second event
}

// --- get_config() Tests ---

// The Python test checked the dict keys; here the serialized form (what the
// CLI prints) is checked for the same keys.
#[test]
fn test_get_config_returns_correct_structure() {
    let config = get_config(&[RTHK_BEEP_PATTERN]);

    let json = serde_json::to_value(&config).unwrap();
    let object = json.as_object().expect("config should serialize to an object");
    assert!(object.contains_key("default_seconds_per_chunk"));
    assert!(object.contains_key("min_chunk_size_seconds"));
    assert!(object.contains_key("sample_rate"));
    assert!(object.contains_key("clips"));
}

// default_seconds_per_chunk always returns the constant value.
#[test]
fn test_get_config_default_seconds_per_chunk() {
    // Default seconds_per_chunk
    let config1 = detector_for(&[RTHK_BEEP_PATTERN], DetectorOptions::default()).get_config();
    assert_eq!(config1.default_seconds_per_chunk, DEFAULT_SECONDS_PER_CHUNK);

    // Custom seconds_per_chunk (should still return the constant as default)
    let options2 = DetectorOptions { seconds_per_chunk: Some(30), ..Default::default() };
    let config2 = detector_for(&[RTHK_BEEP_PATTERN], options2).get_config();
    assert_eq!(config2.default_seconds_per_chunk, DEFAULT_SECONDS_PER_CHUNK);

    // Auto mode (None) (should still return the constant as default)
    let options3 = DetectorOptions { seconds_per_chunk: None, ..Default::default() };
    let config3 = detector_for(&[RTHK_BEEP_PATTERN], options3).get_config();
    assert_eq!(config3.default_seconds_per_chunk, DEFAULT_SECONDS_PER_CHUNK);
}

#[test]
fn test_get_config_sample_rate() {
    let config = get_config(&[RTHK_BEEP_PATTERN]);

    assert_eq!(config.sample_rate, DEFAULT_TARGET_SAMPLE_RATE);
    assert_eq!(config.sample_rate, 8000);
}

#[test]
fn test_get_config_min_chunk_size_single_pattern() {
    let config = get_config(&[RTHK_BEEP_PATTERN]);

    // min_chunk_size should be sliding_window * 2
    let clip_config = config.clip(RTHK_BEEP_NAME).expect("rthk_beep should be in clips");
    let expected_min = clip_config.sliding_window_seconds * 2;
    assert_eq!(config.min_chunk_size_seconds, expected_min);
}

// min_chunk_size_seconds is the max of all patterns' minimums.
#[test]
fn test_get_config_min_chunk_size_multiple_patterns() {
    let pattern_files = [
        RTHK_BEEP_PATTERN,     // Short beep
        CBS_NEWS_PATTERN,      // Longer pattern
        RAINBOW_INTRO_PATTERN, // Another pattern
    ];
    let config = get_config(&pattern_files);

    let mut expected_min = 0;
    for (_clip_name, clip_config) in &config.clips {
        let min_for_clip = clip_config.sliding_window_seconds * 2;
        if min_for_clip > expected_min {
            expected_min = min_for_clip;
        }
    }

    assert_eq!(config.min_chunk_size_seconds, expected_min);
    // The larger patterns should determine the min
    assert!(config.min_chunk_size_seconds >= 2); // At least 2 seconds
}

#[test]
fn test_get_config_clips_info() {
    let config = get_config(&[RTHK_BEEP_PATTERN]);

    let clip_config = config.clip(RTHK_BEEP_NAME).expect("rthk_beep should be in clips");

    // Required fields and their types, as serialized.
    let json = serde_json::to_value(&config).unwrap();
    let clip_json = &json["clips"][RTHK_BEEP_NAME];
    assert!(clip_json["duration_seconds"].is_f64());
    assert!(clip_json["sliding_window_seconds"].is_u64());

    // Reasonable values
    assert!(clip_config.duration_seconds > 0.0);
    assert!(clip_config.sliding_window_seconds >= 1);
}

#[test]
fn test_get_config_clips_multiple_patterns() {
    let pattern_files = [RTHK_BEEP_PATTERN, CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN];
    let config = get_config(&pattern_files);

    // All patterns should be in the clips, in the order given
    assert_eq!(clip_names(&config), vec![RTHK_BEEP_NAME, CBS_NEWS_NAME, RAINBOW_INTRO_NAME]);
    assert!(config.clip(RTHK_BEEP_NAME).is_some());
    assert!(config.clip(CBS_NEWS_NAME).is_some());
    assert!(config.clip(RAINBOW_INTRO_NAME).is_some());
    assert_eq!(config.clips.len(), 3);
}

#[test]
fn test_get_config_clip_duration() {
    let config1 = get_config(&[RTHK_BEEP_PATTERN]);
    assert!(config1.clip(RTHK_BEEP_NAME).unwrap().duration_seconds < 0.5);

    let config2 = get_config(&[RAINBOW_INTRO_PATTERN]);
    assert!(config2.clip(RAINBOW_INTRO_NAME).unwrap().duration_seconds >= 0.5);
}

// sliding_window_seconds is the ceil of the clip duration.
#[test]
fn test_get_config_sliding_window_computed_correctly() {
    for pattern_file in [RTHK_BEEP_PATTERN, CBS_NEWS_PATTERN] {
        let pattern_clip = AudioClip::from_audio_file(pattern_file, SR).unwrap();
        let clip_name = pattern_clip.name.clone();
        let detector = AudioPatternDetector::new(vec![pattern_clip], DetectorOptions::default()).unwrap();
        let config = detector.get_config();

        let clip_config = config.clip(&clip_name).unwrap();

        let expected_sliding_window = clip_config.duration_seconds.ceil() as u32;
        assert_eq!(
            clip_config.sliding_window_seconds, expected_sliding_window,
            "{clip_name}: Expected sliding_window {expected_sliding_window}, got {}",
            clip_config.sliding_window_seconds
        );
    }
}
