//! Sliding window behaviour of `AudioPatternDetector`:
//! 1. Detections after the first window have correct timestamps
//! 2. Detections work properly when patterns are at window boundaries
//! 3. Minimum validation for `seconds_per_chunk`
//! 4. Auto-computation of `seconds_per_chunk` when `None` or `Some(0)`

mod common;

use audio_pattern_detector_core::stream::WavFileSource;
use audio_pattern_detector_core::{
    match_pattern, AudioClip, AudioPatternDetector, AudioStream, DetectorOptions, MatchOptions, PeakTimes,
};
use common::{clip_from_samples, concat, insert_at, silence, sine_tone, stream_from_samples, Rng, SR};

const RTHK_BEEP_PATTERN: &str = "../../sample_audios/clips/rthk_beep.apd.toml";
const RTHK_BEEP_AUDIO: &str = "../../sample_audios/rthk_section_with_beep.wav";
const CBS_NEWS_PATTERN: &str = "../../sample_audios/clips/cbs_news.wav";
const CBS_NEWS_AUDIO: &str = "../../sample_audios/cbs_news_audio_section.wav";

const BEEP_NAME: &str = "test_beep";
const BEEP_DURATION: f64 = 0.23;
const LONG_BEEP_NAME: &str = "long_beep";
const LONG_BEEP_DURATION: f64 = 2.5;
const TONE_FREQUENCY: f64 = 1000.0;

fn options(seconds_per_chunk: Option<u32>) -> DetectorOptions {
    DetectorOptions { debug_mode: false, seconds_per_chunk, ..DetectorOptions::default() }
}

fn tone_clip(name: &str, frequency: f64, duration: f64) -> AudioClip {
    clip_from_samples(name, &sine_tone(frequency, duration, SR))
}

/// Synthetic 1 kHz beep pattern.
fn beep_pattern() -> AudioClip {
    tone_clip(BEEP_NAME, TONE_FREQUENCY, BEEP_DURATION)
}

/// Longer synthetic beep: a 2.5 second pattern gives sliding_window = ceil(2.5) = 3 seconds.
fn long_beep_pattern() -> AudioClip {
    tone_clip(LONG_BEEP_NAME, TONE_FREQUENCY, LONG_BEEP_DURATION)
}

/// silence + pattern + silence, `audio_duration` seconds long.
fn pattern_in_silence(pattern: &AudioClip, pattern_start: f64, pattern_duration: f64, audio_duration: f64) -> Vec<f32> {
    let silence_before = silence(pattern_start, SR);
    let silence_after = silence(audio_duration - pattern_start - pattern_duration, SR);
    concat(&[&silence_before, &pattern.audio, &silence_after])
}

/// Silence of `audio_duration` seconds with the pattern written at each position that fits.
fn patterns_at_positions(pattern: &AudioClip, positions: &[f64], audio_duration: f64) -> Vec<f32> {
    let mut audio = silence(audio_duration, SR);
    for &pos in positions {
        let start_sample = (pos * SR as f64) as usize;
        if start_sample + pattern.audio.len() <= audio.len() {
            insert_at(&mut audio, start_sample, &pattern.audio);
        }
    }
    audio
}

fn truncate_to(mut audio: Vec<f32>, audio_duration: f64) -> Vec<f32> {
    audio.truncate((audio_duration * SR as f64) as usize);
    audio
}

fn detect(pattern: &AudioClip, audio: &[f32], seconds_per_chunk: u32) -> PeakTimes {
    let detector = AudioPatternDetector::new(vec![pattern.clone()], options(Some(seconds_per_chunk))).unwrap();
    let mut stream = stream_from_samples("test_audio", audio);
    let (peak_times, _total_time) = detector.find_clip_in_audio(&mut stream, None, true).unwrap();
    peak_times.expect("results are accumulated")
}

/// Detections for `name`, asserting the clip is present in the results.
fn detections<'a>(peak_times: &'a PeakTimes, name: &str) -> &'a [f64] {
    peak_times
        .get(name)
        .unwrap_or_else(|| panic!("'{name}' missing from results: {peak_times:?}"))
}

fn closest_to(times: &[f64], expected: f64) -> f64 {
    times
        .iter()
        .copied()
        .min_by(|a, b| (a - expected).abs().total_cmp(&(b - expected).abs()))
        .expect("at least one detection")
}

fn new_detector(clips: Vec<AudioClip>, seconds_per_chunk: Option<u32>) -> audio_pattern_detector_core::Result<AudioPatternDetector> {
    AudioPatternDetector::new(clips, options(seconds_per_chunk))
}

fn assert_too_small(clips: Vec<AudioClip>, seconds_per_chunk: u32) {
    match new_detector(clips, Some(seconds_per_chunk)) {
        Ok(_) => panic!("seconds_per_chunk {seconds_per_chunk} should be rejected"),
        Err(err) => assert!(err.to_string().contains("too small"), "unexpected error: {err}"),
    }
}

/// Correct timestamp calculation across sliding windows.
mod sliding_window_timestamps {
    use super::*;

    // First chunk (index=0): subtract_seconds=0, so timestamp = peak_time - clip_seconds.
    #[test]
    fn detection_in_first_chunk_has_correct_timestamp() {
        let pattern = beep_pattern();
        let pattern_start = 1.0;
        let audio = pattern_in_silence(&pattern, pattern_start, BEEP_DURATION, 5.0);

        // Large chunk so everything is in the first chunk.
        let peak_times = detect(&pattern, &audio, 60);

        let times = detections(&peak_times, BEEP_NAME);
        assert_eq!(times.len(), 1, "detections: {times:?}");
        let actual_time = times[0];
        assert!(
            (actual_time - pattern_start).abs() < 0.1,
            "Expected timestamp ~{pattern_start}s, got {actual_time}s"
        );
    }

    // Chunk index=1 with seconds_per_chunk=3:
    // (peak_in_section/sr) - subtract_seconds + (index * seconds_per_chunk) - clip_seconds
    #[test]
    fn detection_in_second_chunk_has_correct_timestamp() {
        let pattern = beep_pattern();
        let pattern_start = 4.0;
        let audio = pattern_in_silence(&pattern, pattern_start, BEEP_DURATION, 10.0);

        let peak_times = detect(&pattern, &audio, 3);

        let times = detections(&peak_times, BEEP_NAME);
        assert!(!times.is_empty(), "Expected at least 1 detection, got {}", times.len());
        let closest_detection = closest_to(times, pattern_start);
        assert!(
            (closest_detection - pattern_start).abs() < 0.2,
            "Expected timestamp ~{pattern_start}s, got {closest_detection}s (all: {times:?})"
        );
    }

    #[test]
    fn detection_in_third_chunk_has_correct_timestamp() {
        let pattern = beep_pattern();
        // 7.0 seconds is in the third chunk (index=2).
        let pattern_start = 7.0;
        let audio = pattern_in_silence(&pattern, pattern_start, BEEP_DURATION, 12.0);

        let peak_times = detect(&pattern, &audio, 3);

        let times = detections(&peak_times, BEEP_NAME);
        assert!(!times.is_empty(), "Expected at least 1 detection, got {}", times.len());
        let closest_detection = closest_to(times, pattern_start);
        assert!(
            (closest_detection - pattern_start).abs() < 0.2,
            "Expected timestamp ~{pattern_start}s, got {closest_detection}s (all: {times:?})"
        );
    }

    #[test]
    fn multiple_detections_across_chunks_have_correct_timestamps() {
        let pattern = beep_pattern();
        // 1.0s (chunk 0), 4.5s (chunk 1), 8.0s (chunk 2)
        let pattern_positions = [1.0, 4.5, 8.0];
        let audio = patterns_at_positions(&pattern, &pattern_positions, 12.0);

        let peak_times = detect(&pattern, &audio, 3);

        let times = detections(&peak_times, BEEP_NAME);
        for expected_pos in pattern_positions {
            assert!(
                times.iter().any(|actual| (actual - expected_pos).abs() < 0.3),
                "No detection found near {expected_pos}s (detections: {times:?})"
            );
        }
    }
}

/// Pattern detection at chunk boundaries.
mod sliding_window_boundary {
    use super::*;

    fn assert_boundary_detection(pattern_start: f64, audio_duration: f64, tolerance: f64, message: &str) {
        let pattern = beep_pattern();
        let audio = pattern_in_silence(&pattern, pattern_start, BEEP_DURATION, audio_duration);

        let peak_times = detect(&pattern, &audio, 3);

        let times = detections(&peak_times, BEEP_NAME);
        assert!(!times.is_empty(), "{message}, got {} detections", times.len());
        let closest_detection = closest_to(times, pattern_start);
        assert!(
            (closest_detection - pattern_start).abs() < tolerance,
            "Expected detection near {pattern_start}s, got {closest_detection}s"
        );
    }

    // First chunk ends at 3.0s; the pattern spans 2.9 to 3.13 seconds.
    #[test]
    fn detection_at_chunk_boundary_is_found() {
        assert_boundary_detection(2.9, 10.0, 0.3, "Pattern at boundary should be detected");
    }

    // Pattern starts exactly at the chunk boundary.
    #[test]
    fn detection_just_after_boundary_has_correct_timestamp() {
        assert_boundary_detection(3.0, 10.0, 0.3, "Pattern just after boundary should be detected");
    }

    // Pattern ends right at the chunk boundary (~2.77s).
    #[test]
    fn detection_just_before_boundary_has_correct_timestamp() {
        assert_boundary_detection(3.0 - BEEP_DURATION, 10.0, 0.3, "Pattern just before boundary should be detected");
    }

    // sliding_window = ceil(0.23) = 1 second, so the overlap region for chunk 2
    // is 2.0s to 3.0s (last 1s of chunk 1). The pattern may be detected in both
    // chunks but should have a consistent timestamp.
    #[test]
    fn sliding_window_overlap_captures_boundary_pattern() {
        assert_boundary_detection(2.5, 10.0, 0.3, "Pattern in overlap region should be detected");
    }

    // Regression: the final-short-chunk branch used to take the last
    // seconds_per_chunk seconds of (previous + chunk) instead of prepending
    // sliding_window. With a final chunk of 2.95s (close to the 3s chunk size)
    // that left only 0.05s of lookback, so a pattern crossing the boundary fell
    // out of bounds in both chunks and was silently dropped.
    #[test]
    fn pattern_straddling_final_short_chunk_boundary_is_found() {
        assert_boundary_detection(
            2.9,
            5.95,
            0.1,
            "Pattern straddling boundary into final short chunk should be detected",
        );
    }
}

/// Each clip's section is loudness-normalized on its own lookback, so one
/// clip's detections never depend on which other clips are loaded.
mod per_clip_normalization {
    use super::*;

    const UNRELATED_NAME: &str = "unrelated_long";

    /// Loud unrelated audio at 6.5-8.5s, then a quiet beep at 11s. With 10s
    /// chunks the beep's own 1s lookback section (9-20s) holds only the
    /// beep, while a 4s lookback (6-20s) would also take in the loud audio.
    fn quiet_beep_after_loud_audio() -> Vec<f32> {
        let mut audio = silence(30.0, SR);
        let loud: Vec<f32> = sine_tone(300.0, 2.0, SR).iter().map(|v| v * 0.9).collect();
        insert_at(&mut audio, (6.5 * SR as f64) as usize, &loud);
        let quiet: Vec<f32> = beep_pattern().audio.iter().map(|v| v * 0.02).collect();
        insert_at(&mut audio, (11.0 * SR as f64) as usize, &quiet);
        audio
    }

    fn beep_detections(clips: Vec<AudioClip>) -> Vec<f64> {
        let detector = new_detector(clips, Some(10)).unwrap();
        let mut stream = stream_from_samples("test_audio", &quiet_beep_after_loud_audio());
        let (peak_times, _) = detector.find_clip_in_audio(&mut stream, None, true).unwrap();
        detections(&peak_times.unwrap(), BEEP_NAME).to_vec()
    }

    #[test]
    fn unrelated_longer_clip_does_not_change_detections() {
        let alone = beep_detections(vec![beep_pattern()]);
        assert_eq!(alone.len(), 1, "quiet beep alone: {alone:?}");
        assert!((alone[0] - 11.0).abs() < 0.05, "quiet beep alone at {alone:?}");

        let unrelated = tone_clip(UNRELATED_NAME, 2000.0, 3.5);
        let with_unrelated = beep_detections(vec![beep_pattern(), unrelated]);
        assert_eq!(with_unrelated, alone, "detections changed when an unrelated clip was added");
    }
}

/// Clips with the same sliding window share one loudness-normalized section
/// and one forward FFT (sized for the longest clip of the group). Sharing
/// must not change any clip's detections compared with running it alone.
mod shared_sections {
    use super::*;

    const MID_NAME: &str = "mid_tone";
    const MID_DURATION: f64 = 0.6;
    const MID_FREQUENCY: f64 = 2000.0;
    const LOW_NAME: &str = "low_long";
    const LOW_FREQUENCY: f64 = 500.0;

    /// 0.6s 2 kHz tone: sliding_window 1 like the beep, but longer, so
    /// the shared FFT is sized for it rather than for the beep.
    fn mid_pattern() -> AudioClip {
        tone_clip(MID_NAME, MID_FREQUENCY, MID_DURATION)
    }

    /// 2.5s 500 Hz tone: sliding_window 3, its own group.
    fn low_pattern() -> AudioClip {
        tone_clip(LOW_NAME, LOW_FREQUENCY, LONG_BEEP_DURATION)
    }

    // 10s chunks: positions in the first chunk, in a later chunk, and
    // straddling a chunk boundary (so the lookback matters).
    const BEEP_POSITIONS: &[f64] = &[3.0, 9.9, 15.0];
    const MID_POSITIONS: &[f64] = &[5.0, 12.0, 19.8];
    const LOW_POSITIONS: &[f64] = &[0.5, 24.0];
    const AUDIO_SECONDS: f64 = 30.0;
    const SECONDS_PER_CHUNK: u32 = 10;

    fn mixed_audio() -> Vec<f32> {
        let mut audio = silence(AUDIO_SECONDS, SR);
        for (pattern, positions) in
            [(beep_pattern(), BEEP_POSITIONS), (mid_pattern(), MID_POSITIONS), (low_pattern(), LOW_POSITIONS)]
        {
            for &pos in positions {
                insert_at(&mut audio, (pos * SR as f64) as usize, &pattern.audio);
            }
        }
        audio
    }

    fn run(clips: Vec<AudioClip>) -> PeakTimes {
        let detector = new_detector(clips, Some(SECONDS_PER_CHUNK)).unwrap();
        let mut stream = stream_from_samples("test_audio", &mixed_audio());
        let (peak_times, _) = detector.find_clip_in_audio(&mut stream, None, true).unwrap();
        peak_times.expect("results are accumulated")
    }

    fn assert_found_at(times: &[f64], positions: &[f64], name: &str) {
        assert_eq!(times.len(), positions.len(), "{name}: {times:?} vs {positions:?}");
        for (actual, expected) in times.iter().zip(positions) {
            assert!((actual - expected).abs() < 0.05, "{name}: {actual} vs {expected} (all: {times:?})");
        }
    }

    #[test]
    fn each_clip_alone_finds_every_occurrence() {
        assert_found_at(detections(&run(vec![beep_pattern()]), BEEP_NAME), BEEP_POSITIONS, BEEP_NAME);
        assert_found_at(detections(&run(vec![mid_pattern()]), MID_NAME), MID_POSITIONS, MID_NAME);
        assert_found_at(detections(&run(vec![low_pattern()]), LOW_NAME), LOW_POSITIONS, LOW_NAME);
    }

    #[test]
    fn clips_sharing_a_section_match_their_solo_runs_exactly() {
        let beep_alone = detections(&run(vec![beep_pattern()]), BEEP_NAME).to_vec();
        let mid_alone = detections(&run(vec![mid_pattern()]), MID_NAME).to_vec();

        // Same group, shorter clip first (FFT sized for the second clip).
        let together = run(vec![beep_pattern(), mid_pattern()]);
        assert_eq!(detections(&together, BEEP_NAME), beep_alone);
        assert_eq!(detections(&together, MID_NAME), mid_alone);

        // Same group, longer clip first.
        let together = run(vec![mid_pattern(), beep_pattern()]);
        assert_eq!(detections(&together, BEEP_NAME), beep_alone);
        assert_eq!(detections(&together, MID_NAME), mid_alone);
    }

    #[test]
    fn clips_in_different_groups_match_their_solo_runs_exactly() {
        let beep_alone = detections(&run(vec![beep_pattern()]), BEEP_NAME).to_vec();
        let mid_alone = detections(&run(vec![mid_pattern()]), MID_NAME).to_vec();
        let low_alone = detections(&run(vec![low_pattern()]), LOW_NAME).to_vec();

        // Groups in both orders of first appearance.
        for clips in [
            vec![beep_pattern(), low_pattern(), mid_pattern()],
            vec![low_pattern(), mid_pattern(), beep_pattern()],
        ] {
            let together = run(clips);
            assert_eq!(detections(&together, BEEP_NAME), beep_alone);
            assert_eq!(detections(&together, MID_NAME), mid_alone);
            assert_eq!(detections(&together, LOW_NAME), low_alone);
        }
    }

    #[test]
    fn duplicate_clips_get_identical_detections() {
        let mut copy = beep_pattern();
        copy.name = "beep_copy".to_string();
        let together = run(vec![beep_pattern(), copy]);
        let beep = detections(&together, BEEP_NAME);
        assert_found_at(beep, BEEP_POSITIONS, BEEP_NAME);
        assert_eq!(detections(&together, "beep_copy"), beep);
    }

    #[test]
    fn callback_order_is_by_time_within_a_chunk_across_groups() {
        use std::sync::{Arc, Mutex};
        let clips = vec![beep_pattern(), low_pattern(), mid_pattern()];
        let detector = new_detector(clips, Some(SECONDS_PER_CHUNK)).unwrap();
        let mut stream = stream_from_samples("test_audio", &mixed_audio());
        let events: Arc<Mutex<Vec<(f64, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let mut callback = move |name: &str, time: f64| sink.lock().unwrap().push((time, name.to_string()));
        let (peak_times, _) = detector.find_clip_in_audio(&mut stream, Some(&mut callback), true).unwrap();
        let peak_times = peak_times.unwrap();

        let events = events.lock().unwrap();
        let times: Vec<f64> = events.iter().map(|(t, _)| *t).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]), "callback not in time order: {events:?}");

        // The callback saw exactly the accumulated detections of every clip.
        for name in [BEEP_NAME, MID_NAME, LOW_NAME] {
            let from_callback: Vec<f64> = events.iter().filter(|(_, n)| n == name).map(|(t, _)| *t).collect();
            assert_eq!(from_callback, detections(&peak_times, name), "{name}");
        }
    }
}

/// Integration tests using real audio patterns for sliding window behaviour.
mod sliding_window_with_real_patterns {
    use super::*;

    fn detect_in_wav(pattern_file: &str, audio_file: &str, seconds_per_chunk: u32) -> PeakTimes {
        let pattern_clip = AudioClip::from_audio_file(pattern_file, SR).unwrap();
        let source = WavFileSource::open(audio_file, SR).unwrap();
        let mut stream = AudioStream::new("audio", source, SR);
        let detector = new_detector(vec![pattern_clip], Some(seconds_per_chunk)).unwrap();
        let (peak_times, _) = detector.find_clip_in_audio(&mut stream, None, true).unwrap();
        peak_times.expect("results are accumulated")
    }

    // Small chunks must not affect timestamp accuracy.
    #[test]
    fn rthk_beep_detection_with_small_chunks() {
        // Reference results with the default chunk size.
        let (reference_results, _) = match_pattern(
            RTHK_BEEP_AUDIO,
            &[RTHK_BEEP_PATTERN],
            &MatchOptions { debug_mode: false, ..MatchOptions::default() },
            None,
            true,
        )
        .unwrap();
        let reference_results = reference_results.expect("results are accumulated");

        let small_chunk_results = detect_in_wav(RTHK_BEEP_PATTERN, RTHK_BEEP_AUDIO, 3);

        assert_eq!(detections(&reference_results, "rthk_beep").len(), 2);
        let small_chunk_times = detections(&small_chunk_results, "rthk_beep");
        assert!(small_chunk_times.len() >= 2, "detections: {small_chunk_times:?}");

        let expected_times = [1.4165, 2.419125];
        for expected in expected_times {
            assert!(
                small_chunk_times.iter().any(|actual| (actual - expected).abs() < 0.1),
                "Expected detection near {expected}s not found in {small_chunk_times:?}"
            );
        }
    }

    // The CBS news pattern is detected at ~25.9s, which is chunk index=2 with
    // 10-second chunks. A pattern in the overlap region may be detected in both
    // chunks, so duplicates are expected sliding window behaviour.
    #[test]
    fn cbs_news_detection_with_multiple_chunks() {
        let peak_times = detect_in_wav(CBS_NEWS_PATTERN, CBS_NEWS_AUDIO, 10);

        let times = detections(&peak_times, "cbs_news");
        assert!(!times.is_empty());

        let expected_time = 25.89875;
        let closest = closest_to(times, expected_time);
        assert!(
            (closest - expected_time).abs() < 0.1,
            "Expected timestamp ~{expected_time}s, got {closest}s (all: {times:?})"
        );
    }
}

/// Edge cases in timestamp calculation.
mod timestamp_calculation_edge_cases {
    use super::*;

    #[test]
    fn pattern_at_very_beginning_of_audio() {
        let pattern = beep_pattern();
        let audio_duration = 5.0;
        let silence_after = silence(audio_duration - BEEP_DURATION, SR);
        let audio = concat(&[&pattern.audio, &silence_after]);

        let peak_times = detect(&pattern, &audio, 60);

        let times = detections(&peak_times, BEEP_NAME);
        if let Some(&first) = times.first() {
            assert!(first >= 0.0, "Timestamp should not be negative: {first}");
            assert!(first < 0.5, "Detection at beginning should have small timestamp: {first}");
        }
    }

    // Audio is 8.5 seconds, so the last chunk is partial.
    #[test]
    fn pattern_near_end_of_last_chunk() {
        let pattern = beep_pattern();
        let audio_duration = 8.5;
        let pattern_start = audio_duration - BEEP_DURATION - 0.1;
        let silence_before = silence(pattern_start, SR);
        let audio = truncate_to(concat(&[&silence_before, &pattern.audio]), audio_duration);

        let peak_times = detect(&pattern, &audio, 3);

        let times = detections(&peak_times, BEEP_NAME);
        if !times.is_empty() {
            let closest = closest_to(times, pattern_start);
            assert!(
                (closest - pattern_start).abs() < 0.5,
                "Expected detection near {pattern_start}s, got {closest}s"
            );
        }
    }

    // Patterns in the overlap region between chunks may be detected twice;
    // after deduplication timestamps should be monotonically increasing.
    #[test]
    fn timestamps_increase_monotonically_for_sequential_patterns() {
        let pattern = beep_pattern();
        let pattern_positions = [0.5, 2.0, 4.0, 6.5, 9.0];
        let audio = patterns_at_positions(&pattern, &pattern_positions, 12.0);

        let peak_times = detect(&pattern, &audio, 3);

        // Deduplicate timestamps that are very close together (within 0.01s).
        let mut sorted = detections(&peak_times, BEEP_NAME).to_vec();
        sorted.sort_by(f64::total_cmp);
        let mut deduplicated: Vec<f64> = Vec::new();
        for t in sorted {
            if deduplicated.last().is_none_or(|last| (t - last).abs() > 0.01) {
                deduplicated.push(t);
            }
        }

        for pair in deduplicated.windows(2) {
            assert!(pair[1] > pair[0], "Timestamps should be increasing after dedup: {deduplicated:?}");
        }

        let found_count = pattern_positions
            .iter()
            .filter(|&&expected| deduplicated.iter().any(|actual| (actual - expected).abs() < 0.3))
            .count();
        assert!(
            found_count >= pattern_positions.len() - 1,
            "Expected to find most patterns. Positions: {pattern_positions:?}, Detections: {deduplicated:?}"
        );
    }
}

/// Longer patterns (2.5+ seconds) give larger sliding windows (3+ seconds);
/// these catch timestamp drift accumulating across multiple chunks.
mod large_sliding_window {
    use super::*;

    fn assert_single_long_beep(pattern_start: f64, audio_duration: f64, tolerance: f64) {
        let pattern = long_beep_pattern();
        let audio = pattern_in_silence(&pattern, pattern_start, LONG_BEEP_DURATION, audio_duration);

        // 10s chunks (must be >= 2 * sliding_window = 6s).
        let peak_times = detect(&pattern, &audio, 10);

        let times = detections(&peak_times, LONG_BEEP_NAME);
        assert!(!times.is_empty(), "Expected detection, got {}", times.len());
        let closest = closest_to(times, pattern_start);
        assert!(
            (closest - pattern_start).abs() < tolerance,
            "Expected ~{pattern_start}s, got {closest}s (drift detected!)"
        );
    }

    // 12.0s is in the second chunk (index=1).
    #[test]
    fn large_window_detection_in_second_chunk() {
        assert_single_long_beep(12.0, 30.0, 0.5);
    }

    // 45.0s with 10s chunks is chunk index=4.
    #[test]
    fn large_window_detection_in_fifth_chunk() {
        assert_single_long_beep(45.0, 60.0, 0.5);
    }

    #[test]
    fn large_window_multiple_patterns_no_drift() {
        let pattern = long_beep_pattern();
        // 5s (chunk 0), 15s (chunk 1), 35s (chunk 3), 55s (chunk 5)
        let pattern_positions = [5.0, 15.0, 35.0, 55.0];
        let audio = patterns_at_positions(&pattern, &pattern_positions, 70.0);

        let peak_times = detect(&pattern, &audio, 10);

        let times = detections(&peak_times, LONG_BEEP_NAME);
        for expected_pos in pattern_positions {
            assert!(
                times.iter().any(|actual| (actual - expected_pos).abs() < 0.5),
                "No detection near {expected_pos}s (detections: {times:?})"
            );
        }

        let last_expected = pattern_positions[pattern_positions.len() - 1];
        let closest_to_last = closest_to(times, last_expected);
        assert!(
            (closest_to_last - last_expected).abs() < 0.5,
            "Drift detected at end: expected ~{last_expected}s, got {closest_to_last}s"
        );
    }

    // Pattern from 8.5s to 11.0s straddles the chunk boundary at 10s.
    #[test]
    fn large_window_boundary_detection() {
        assert_single_long_beep(8.5, 30.0, 0.5);
    }

    // Pattern duration 4.5s -> sliding_window 5s; stresses the timestamp
    // calculation with large subtract_seconds values.
    #[test]
    fn very_large_window_far_into_audio() {
        let pattern_duration = 4.5;
        let name = "very_long_beep";
        let pattern = tone_clip(name, TONE_FREQUENCY, pattern_duration);
        let pattern_start = 50.0;
        let audio = pattern_in_silence(&pattern, pattern_start, pattern_duration, 70.0);

        // Must be >= 2 * sliding_window = 10s.
        let peak_times = detect(&pattern, &audio, 15);

        let times = detections(&peak_times, name);
        assert!(!times.is_empty());
        let closest = closest_to(times, pattern_start);
        assert!(
            (closest - pattern_start).abs() < 1.0,
            "Expected ~{pattern_start}s, got {closest}s (drift with very large window!)"
        );
    }

    // 95.0s is in the 10th chunk (index=9).
    #[test]
    fn large_window_timestamp_accuracy_across_ten_chunks() {
        assert_single_long_beep(95.0, 110.0, 1.0);
    }

    // With drift the later chunk would have more error than the first.
    #[test]
    fn compare_first_and_tenth_chunk_accuracy() {
        let pattern = long_beep_pattern();
        let early_position = 5.0;
        let late_position = 95.0;
        let mut audio = silence(110.0, SR);
        insert_at(&mut audio, (early_position * SR as f64) as usize, &pattern.audio);
        insert_at(&mut audio, (late_position * SR as f64) as usize, &pattern.audio);

        let peak_times = detect(&pattern, &audio, 10);

        let times = detections(&peak_times, LONG_BEEP_NAME);
        assert!(times.len() >= 2, "detections: {times:?}");

        let first_near = |position: f64| times.iter().copied().find(|t| (t - position).abs() < 1.0);
        let early = first_near(early_position)
            .unwrap_or_else(|| panic!("No detection near early position {early_position}s"));
        let early_error = (early - early_position).abs();
        let late =
            first_near(late_position).unwrap_or_else(|| panic!("No detection near late position {late_position}s"));
        let late_error = (late - late_position).abs();

        assert!(
            (late_error - early_error).abs() < 0.5,
            "Drift detected: early_error={early_error:.3}s, late_error={late_error:.3}s"
        );
    }
}

/// A pattern in the overlap region may be detected by both the current chunk
/// and the next one; both detections should report the same timestamp.
mod sliding_window_overlap_deduplication {
    use super::*;

    // sliding_window = ceil(3.5) = 4
    const OVERLAP_PATTERN_DURATION: f64 = 3.5;
    const OVERLAP_SECONDS_PER_CHUNK: u32 = 10;

    fn overlap_pattern(name: &str) -> AudioClip {
        tone_clip(name, TONE_FREQUENCY, OVERLAP_PATTERN_DURATION)
    }

    // Chunk 0 processes 0-10s and detects the pattern at ~7s; chunk 1
    // processes 6-20s (4s overlap) and may also detect it at ~7s.
    #[test]
    fn pattern_in_overlap_detected_with_same_timestamp() {
        let name = "overlap_test";
        let pattern = overlap_pattern(name);
        let pattern_start = 7.0;
        let audio = pattern_in_silence(&pattern, pattern_start, OVERLAP_PATTERN_DURATION, 20.0);

        let peak_times = detect(&pattern, &audio, OVERLAP_SECONDS_PER_CHUNK);

        let times = detections(&peak_times, name);
        println!("Detections: {times:?}");
        let closest = closest_to(times, pattern_start);
        assert!(
            (closest - pattern_start).abs() < 0.5,
            "Expected detection near {pattern_start}s, got {closest}s"
        );
    }

    // Pattern at 8s is definitely in the overlap (6-10s of chunk 0).
    #[test]
    fn overlap_duplicate_timestamps_are_identical() {
        let name = "dedup_test";
        let pattern = overlap_pattern(name);
        let pattern_start = 8.0;
        let audio = pattern_in_silence(&pattern, pattern_start, OVERLAP_PATTERN_DURATION, 25.0);

        let peak_times = detect(&pattern, &audio, OVERLAP_SECONDS_PER_CHUNK);

        let times = detections(&peak_times, name);
        println!("Detections for dedup_test: {times:?}");
        if times.len() > 1 {
            for t in times {
                assert!(
                    (t - pattern_start).abs() < 0.5,
                    "Detection {t}s too far from expected {pattern_start}s"
                );
            }
        }
    }

    // Pattern ends exactly at the 10s chunk boundary.
    #[test]
    fn pattern_exactly_at_chunk_boundary_overlap() {
        let name = "boundary_exact";
        let pattern = overlap_pattern(name);
        let pattern_start = 10.0 - OVERLAP_PATTERN_DURATION; // 6.5s
        let audio = pattern_in_silence(&pattern, pattern_start, OVERLAP_PATTERN_DURATION, 25.0);

        let peak_times = detect(&pattern, &audio, OVERLAP_SECONDS_PER_CHUNK);

        let times = detections(&peak_times, name);
        println!("Boundary exact detections: {times:?}");
        assert!(!times.is_empty(), "Pattern at boundary should be detected");
        for t in times {
            assert!(
                (t - pattern_start).abs() < 0.5,
                "Detection {t}s too far from expected {pattern_start}s"
            );
        }
    }

    // Audio: 20s, chunks: 10s, pattern at 9s (extends to 12.5s, crossing into
    // chunk 1). The Python test only printed the detections, so this checks
    // that detection runs and reports the clip.
    #[test]
    fn short_pattern_large_sliding_window_scenario() {
        let name = "user_scenario";
        let pattern = overlap_pattern(name);
        let pattern_start = 9.0;
        let audio = pattern_in_silence(&pattern, pattern_start, OVERLAP_PATTERN_DURATION, 20.0);

        let peak_times = detect(&pattern, &audio, OVERLAP_SECONDS_PER_CHUNK);

        let times = detections(&peak_times, name);
        println!("User scenario detections: {times:?}");
    }

    // For a pattern at absolute position P with duration D:
    // - Chunk 0 (index=0): final_ts = (P + D) - 0 + 0 - D = P
    // - Chunk 1 (index=1) with sliding_window S: the pattern is at
    //   (P + D - (chunk_size - S)) in the section, so
    //   final_ts = (P + D - (chunk_size - S)) - S + chunk_size - D = P
    #[test]
    fn verify_duplicate_timestamp_calculation() {
        let name = "calc_verify";
        let pattern = overlap_pattern(name);
        let test_positions = [6.5, 7.0, 8.0, 9.0];
        let audio_duration = 25.0;

        for pattern_start in test_positions {
            let silence_before = silence(pattern_start, SR);
            let remaining: f64 = audio_duration - pattern_start - OVERLAP_PATTERN_DURATION;
            let silence_after = silence(remaining.max(0.0), SR);
            let audio = truncate_to(concat(&[&silence_before, &pattern.audio, &silence_after]), audio_duration);

            let peak_times = detect(&pattern, &audio, OVERLAP_SECONDS_PER_CHUNK);

            let times = detections(&peak_times, name);
            for t in times {
                let error = (t - pattern_start).abs();
                assert!(
                    error < 0.5,
                    "Pattern at {pattern_start}s: detection at {t}s has error {error:.3}s"
                );
            }
            for (i, t1) in times.iter().enumerate() {
                for t2 in &times[i + 1..] {
                    let diff = (t1 - t2).abs();
                    assert!(diff < 0.1, "Duplicate timestamps differ: {t1}s vs {t2}s (diff={diff:.4}s)");
                }
            }
        }
    }
}

/// Validation rules:
/// 1. seconds_per_chunk must be >= 2 * sliding_window (error from `new`)
/// 2. `None` or `Some(0)` auto-computes it as longest_clip * 2
mod seconds_per_chunk_validation {
    use super::*;

    const PATTERN_NAME: &str = "test_pattern";

    // 2.5s -> sliding_window = ceil(2.5) = 3s
    fn test_pattern() -> AudioClip {
        tone_clip(PATTERN_NAME, TONE_FREQUENCY, 2.5)
    }

    fn short_and_long_patterns() -> Vec<AudioClip> {
        // Short: sliding_window = 1s -> min chunk = 2s; long: sliding_window = 3s -> min chunk = 6s.
        vec![tone_clip("short", TONE_FREQUENCY, 0.5), tone_clip("long", 500.0, 3.0)]
    }

    // 5 < 2 * 3 = 6
    #[test]
    fn seconds_per_chunk_too_small_raises_error() {
        assert_too_small(vec![test_pattern()], 5);
    }

    // 6 = 2 * 3, exactly at minimum
    #[test]
    fn seconds_per_chunk_exactly_minimum_works() {
        let detector = new_detector(vec![test_pattern()], Some(6)).unwrap();
        assert_eq!(detector.seconds_per_chunk(), 6);
    }

    #[test]
    fn seconds_per_chunk_above_minimum_works() {
        let detector = new_detector(vec![test_pattern()], Some(10)).unwrap();
        assert_eq!(detector.seconds_per_chunk(), 10);
    }

    // Auto-computed: ceil(clip_length_samples / sr) * 2 = ceil(2.5) * 2 = 6
    #[test]
    fn seconds_per_chunk_none_auto_computes() {
        let detector = new_detector(vec![test_pattern()], None).unwrap();
        assert_eq!(detector.seconds_per_chunk(), 6);
    }

    #[test]
    fn seconds_per_chunk_zero_auto_computes() {
        let detector = new_detector(vec![test_pattern()], Some(0)).unwrap();
        assert_eq!(detector.seconds_per_chunk(), 6);
    }

    // seconds_per_chunk=4 fails because 4 < 6 (for the longest pattern).
    #[test]
    fn multiple_patterns_uses_longest_for_validation() {
        assert_too_small(short_and_long_patterns(), 4);
    }

    // 8 > 6 (2 * 3)
    #[test]
    fn multiple_patterns_valid_chunk_size() {
        let detector = new_detector(short_and_long_patterns(), Some(8)).unwrap();
        assert_eq!(detector.seconds_per_chunk(), 8);
    }

    // A 0.23s pattern has sliding_window = 1s, so 2s works (exactly 2x).
    #[test]
    fn short_pattern_small_chunk_works() {
        let detector = new_detector(vec![tone_clip("beep", TONE_FREQUENCY, 0.23)], Some(2)).unwrap();
        assert_eq!(detector.seconds_per_chunk(), 2);
    }

    // A 0.5s pattern has sliding_window = 1s; 1 < 2 * 1 = 2.
    #[test]
    fn short_pattern_chunk_just_below_minimum_fails() {
        assert_too_small(vec![tone_clip("short_beep", TONE_FREQUENCY, 0.5)], 1);
    }
}

/// sliding_window computation from pattern duration.
mod sliding_window_computation {
    use super::*;

    // Tested indirectly by checking which chunk sizes work/fail.
    #[test]
    fn sliding_window_is_ceiling_of_pattern_duration() {
        // (pattern_duration, expected_sliding_window)
        let test_cases: [(f64, u32); 7] = [(0.1, 1), (0.5, 1), (1.0, 1), (1.1, 2), (2.0, 2), (2.5, 3), (4.9, 5)];

        for (pattern_duration, expected_sliding_window) in test_cases {
            let pattern = tone_clip("test", TONE_FREQUENCY, pattern_duration);
            let min_valid_chunk = 2 * expected_sliding_window;

            let detector = new_detector(vec![pattern.clone()], Some(min_valid_chunk)).unwrap_or_else(|err| {
                panic!("Pattern {pattern_duration}s: expected chunk {min_valid_chunk}s to work: {err}")
            });
            assert_eq!(detector.seconds_per_chunk(), min_valid_chunk);

            // One less must fail (never 0 here, which would auto-compute).
            assert!(min_valid_chunk > 1);
            assert_too_small(vec![pattern], min_valid_chunk - 1);
        }
    }

    // Longest is 2.5s = 20000 samples -> ceil(20000 / 8000) = 3 -> * 2 = 6
    #[test]
    fn auto_compute_uses_longest_pattern() {
        let patterns = vec![
            tone_clip("p1", TONE_FREQUENCY, 1.0),
            tone_clip("p2", 800.0, 2.5),
            tone_clip("p3", 600.0, 0.3),
        ];

        let detector = new_detector(patterns, None).unwrap();
        assert_eq!(detector.seconds_per_chunk(), 6);
    }
}

const SHORT_BEEP_NAME: &str = "short_beep";
const SHORT_BEEP_DURATION: f64 = 0.1;
/// Five beeps in the last second of the first 2-second chunk, all inside
/// the 1-second lookback of the next chunk.
const OVERLAP_BEEP_POSITIONS: [f64; 5] = [1.05, 1.25, 1.45, 1.65, 1.85];
const OVERLAP_BEEP_EXPECTED_TIMES: [f64; 5] = [1.049875, 1.2498749999999998, 1.4498749999999998, 1.649875, 1.849875];

#[test]
fn test_overlap_detections_are_reported_once() {
    let pattern = tone_clip(SHORT_BEEP_NAME, TONE_FREQUENCY, SHORT_BEEP_DURATION);
    let audio = patterns_at_positions(&pattern, &OVERLAP_BEEP_POSITIONS, 4.0);
    let detector = new_detector(vec![pattern], Some(2)).unwrap();

    let mut events: Vec<(String, f64)> = Vec::new();
    let mut callback = |name: &str, timestamp: f64| events.push((name.to_string(), timestamp));
    let (peak_times, total_time) = detector
        .find_clip_in_audio(&mut stream_from_samples("overlap", &audio), Some(&mut callback), true)
        .unwrap();

    let expected: Vec<f64> = OVERLAP_BEEP_EXPECTED_TIMES.to_vec();
    assert_eq!(total_time, 4.0);
    assert_eq!(detections(&peak_times.unwrap(), SHORT_BEEP_NAME), expected.as_slice());
    let expected_events: Vec<(String, f64)> =
        expected.iter().map(|&t| (SHORT_BEEP_NAME.to_string(), t)).collect();
    assert_eq!(events, expected_events);
}

const TIE_OTHER_NAME: &str = "tie_other";
const TIE_PREFIX_NAME: &str = "tie_prefix";
const TIE_FULL_NAME: &str = "tie_full";
const TIE_PREFIX_DURATION: f64 = 0.5;
const TIE_FULL_DURATION: f64 = 1.5;
const TIE_START: f64 = 2.0;
const TIE_EXPECTED_TIME: f64 = 1.9998749999999998;

// Clips are grouped by sliding window for matching, but two clips detected
// at the same timestamp are still reported in clip order.
#[test]
fn test_same_timestamp_detections_are_reported_in_clip_order() {
    let full_audio = Rng::new(11).noise((TIE_FULL_DURATION * SR as f64) as usize, 0.1);
    let prefix_audio = &full_audio[..(TIE_PREFIX_DURATION * SR as f64) as usize];
    // Sliding windows 2, 1, 2: the prefix clip is matched after both others.
    let clips = vec![
        clip_from_samples(TIE_OTHER_NAME, &Rng::new(12).noise(full_audio.len(), 0.1)),
        clip_from_samples(TIE_PREFIX_NAME, prefix_audio),
        clip_from_samples(TIE_FULL_NAME, &full_audio),
    ];
    let mut audio = silence(6.0, SR);
    insert_at(&mut audio, (TIE_START * SR as f64) as usize, &full_audio);
    let detector = new_detector(clips, Some(10)).unwrap();

    let mut events: Vec<(String, f64)> = Vec::new();
    let mut callback = |name: &str, timestamp: f64| events.push((name.to_string(), timestamp));
    detector
        .find_clip_in_audio(&mut stream_from_samples("tie", &audio), Some(&mut callback), false)
        .unwrap();

    assert_eq!(
        events,
        vec![(TIE_PREFIX_NAME.to_string(), TIE_EXPECTED_TIME), (TIE_FULL_NAME.to_string(), TIE_EXPECTED_TIME)]
    );
}

const THRESHOLD_NAME: &str = "threshold_pattern";
const THRESHOLD_OTHER_NAME: &str = "threshold_other";
const THRESHOLD_PATTERN_SAMPLES: usize = 100;
const THRESHOLD_OTHER_SAMPLES: usize = 5000;
/// Amplitude of the quiet copy that puts its correlation peak right at the
/// default `height_min` (0.25) relative to the full-strength copy.
const THRESHOLD_QUIET_SCALE: f32 = 0.2139812;
const THRESHOLD_QUIET_START: usize = 4000;
const THRESHOLD_FULL_START: usize = 12000;
const THRESHOLD_EXPECTED_TIMES: [f64; 2] = [0.499875, 1.499875];

/// A clip's FFT size depends only on its own length: an unrelated longer
/// clip with the same sliding window (which alone would need a larger FFT:
/// 16000 + 99 <= 16384 < 16000 + 4999) must not change the f32 rounding of
/// a peak sitting right at the height threshold.
#[test]
fn test_threshold_peak_is_independent_of_other_clips_in_the_group() {
    let pattern = Rng::new(0).noise(THRESHOLD_PATTERN_SAMPLES, 0.1);
    let other = Rng::new(1000).noise(THRESHOLD_OTHER_SAMPLES, 0.1);
    let quiet: Vec<f32> = pattern.iter().map(|v| v * THRESHOLD_QUIET_SCALE).collect();
    let mut audio = silence(2.0, SR);
    insert_at(&mut audio, THRESHOLD_QUIET_START, &quiet);
    insert_at(&mut audio, THRESHOLD_FULL_START, &pattern);

    let run = |clips: Vec<AudioClip>| {
        let detector = new_detector(clips, Some(2)).unwrap();
        let (peak_times, _) = detector
            .find_clip_in_audio(&mut stream_from_samples("threshold", &audio), None, true)
            .unwrap();
        detections(&peak_times.unwrap(), THRESHOLD_NAME).to_vec()
    };

    let alone = run(vec![clip_from_samples(THRESHOLD_NAME, &pattern)]);
    let with_other = run(vec![
        clip_from_samples(THRESHOLD_NAME, &pattern),
        clip_from_samples(THRESHOLD_OTHER_NAME, &other),
    ]);
    assert_eq!(alone, THRESHOLD_EXPECTED_TIMES);
    assert_eq!(with_other, THRESHOLD_EXPECTED_TIMES);
}
