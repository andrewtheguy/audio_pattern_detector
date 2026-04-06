import os
import tempfile
from pathlib import Path

import pytest

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import ffmpeg_get_float32_pcm, load_wave_file, write_wav_file, DEFAULT_TARGET_SAMPLE_RATE
from audio_pattern_detector.match import match_pattern, _WavFileStreamWrapper


# --- Test Data Constants ---
# Centralised paths so swapping a clip only requires editing one place.

CBS_NEWS_PATTERN = "sample_audios/clips/cbs_news.wav"
CBS_NEWS_AUDIO = "sample_audios/cbs_news_audio_section.wav"
CBS_NEWS_EXPECTED_TIME = 25.89875

RTHK_BEEP_PATTERN = "sample_audios/clips/rthk_beep.wav"
RTHK_BEEP_AUDIO = "sample_audios/rthk_section_with_beep.wav"
RTHK_BEEP_EXPECTED_TIMES = [1.4165, 2.419125]

RAINBOW_INTRO_PATTERN = "sample_audios/clips/天空下的彩虹intro.wav"
RAINBOW_INTRO_AUDIO = "sample_audios/am1430_section_with_rainbow_intro.wav"
RAINBOW_INTRO_EXPECTED_TIME = 15.5


# --- Pattern Matching Tests ---


def test_rthk_beep_pattern_detection():
    """Test detection of RTHK beep pattern (pure tone/beep detection)

    This tests the beep detection algorithm (_get_peak_times_beep_v3)
    which uses overlap_ratio and downsampled correlation matching.

    Expected to find matches at approximately 1.4165s and 2.419125s
    """
    # Verify input files exist
    assert Path(RTHK_BEEP_PATTERN).exists(), f"Pattern file {RTHK_BEEP_PATTERN} not found"
    assert Path(RTHK_BEEP_AUDIO).exists(), f"Audio file {RTHK_BEEP_AUDIO} not found"

    # Run pattern matching
    peak_times, total_time = match_pattern(RTHK_BEEP_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

    # Verify results structure
    assert isinstance(peak_times, dict), "peak_times should be a dictionary"
    assert 'rthk_beep' in peak_times, "rthk_beep pattern not found in results"

    matches = peak_times['rthk_beep']

    # Verify we found 2 matches
    assert len(matches) == 2, f"Expected 2 matches, found {len(matches)}: {matches}"

    # Verify the timestamps (with tolerance for floating point comparison)
    for i, (actual, expected) in enumerate(zip(sorted(matches), RTHK_BEEP_EXPECTED_TIMES)):
        assert abs(actual - expected) < 0.01, \
            f"Match {i}: Expected timestamp ~{expected}s, got {actual}s"

    # Verify processing time is reasonable
    assert total_time > 0, "Total processing time should be positive"
    assert total_time < 10, f"Processing took too long: {total_time}s"


def test_cbs_news_pattern_detection():
    """Test detection of CBS News pattern (normal audio pattern)

    This tests the normal pattern detection algorithm (_get_peak_times_normal)
    which uses mean squared error and area overlap ratio.

    Expected to find match at approximately 25.89875s
    """
    # Verify input files exist
    assert Path(CBS_NEWS_PATTERN).exists(), f"Pattern file {CBS_NEWS_PATTERN} not found"
    assert Path(CBS_NEWS_AUDIO).exists(), f"Audio file {CBS_NEWS_AUDIO} not found"

    # Run pattern matching
    peak_times, total_time = match_pattern(CBS_NEWS_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)

    # Verify results structure
    assert isinstance(peak_times, dict), "peak_times should be a dictionary"
    assert 'cbs_news' in peak_times, "cbs_news pattern not found in results"

    matches = peak_times['cbs_news']

    # Verify we found 1 match
    assert len(matches) == 1, f"Expected 1 match, found {len(matches)}: {matches}"

    # Verify the timestamp (with tolerance for floating point comparison)
    actual_time = matches[0]
    assert abs(actual_time - CBS_NEWS_EXPECTED_TIME) < 0.01, \
        f"Expected timestamp ~{CBS_NEWS_EXPECTED_TIME}s, got {actual_time}s"

    # Verify processing time is reasonable
    assert total_time > 0, "Total processing time should be positive"


def test_multiple_patterns_detection():
    """Test detection of multiple different normal patterns

    Tests both CBS news and rainbow intro patterns against their
    respective audio files.
    """
    # CBS news pattern
    peak_times_cbs, _ = match_pattern(CBS_NEWS_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)
    assert len(peak_times_cbs['cbs_news']) == 1
    assert abs(peak_times_cbs['cbs_news'][0] - CBS_NEWS_EXPECTED_TIME) < 0.01

    # Rainbow intro pattern
    peak_times_rainbow, _ = match_pattern(RAINBOW_INTRO_AUDIO, [RAINBOW_INTRO_PATTERN], debug_mode=False)
    assert len(peak_times_rainbow['天空下的彩虹intro']) == 1
    assert abs(peak_times_rainbow['天空下的彩虹intro'][0] - RAINBOW_INTRO_EXPECTED_TIME) < 1.0


def test_pattern_not_in_audio():
    """Test that pattern matching returns empty results when pattern is not present

    Tests the algorithm's ability to correctly reject false positives.
    """
    # Use CBS news pattern on RTHK audio (should not match)
    # Verify files exist
    assert Path(CBS_NEWS_PATTERN).exists(), f"Pattern file {CBS_NEWS_PATTERN} not found"
    assert Path(RTHK_BEEP_AUDIO).exists(), f"Audio file {RTHK_BEEP_AUDIO} not found"

    # Run pattern matching
    peak_times, total_time = match_pattern(RTHK_BEEP_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)

    # Should return empty results (no matches)
    assert 'cbs_news' in peak_times, "cbs_news key should exist in results"
    assert len(peak_times['cbs_news']) == 0, \
        f"Expected no matches, but found {len(peak_times['cbs_news'])}: {peak_times['cbs_news']}"


def test_nonexistent_pattern_file():
    """Test that match_pattern raises ValueError for nonexistent pattern file"""
    pattern_file = "sample_audios/clips/nonexistent.wav"

    with pytest.raises(ValueError, match="does not exist"):
        match_pattern(RTHK_BEEP_AUDIO, [pattern_file], debug_mode=False)


def test_nonexistent_audio_file():
    """Test that match_pattern raises ValueError for nonexistent audio file"""
    audio_file = "sample_audios/nonexistent.wav"

    with pytest.raises(ValueError, match="does not exist"):
        match_pattern(audio_file, [RTHK_BEEP_PATTERN], debug_mode=False)


def test_empty_pattern_list():
    """Test that match_pattern raises ValueError when no patterns provided"""
    with pytest.raises(ValueError, match="No pattern clips passed"):
        match_pattern(RTHK_BEEP_AUDIO, [], debug_mode=False)


def test_beep_detection_algorithm_specifics():
    """Test specific behavior of beep detection algorithm

    The beep detection algorithm uses:
    - Downsampled correlation matching
    - Overlap ratio threshold (>0.98 or >0.99 depending on similarity)
    - MSE similarity threshold (<0.01)

    This test verifies the algorithm correctly identifies pure tone patterns.
    """
    peak_times, total_time = match_pattern(RTHK_BEEP_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

    matches = peak_times['rthk_beep']

    # The algorithm should find exactly 2 beeps
    assert len(matches) == 2, "Beep algorithm should find exactly 2 matches"

    # Matches should be in chronological order
    assert matches[0] < matches[1], "Matches should be sorted chronologically"

    # Matches should be separated by a reasonable time
    time_diff = matches[1] - matches[0]
    assert 0.5 < time_diff < 5.0, \
        f"Beeps should be 0.5-5s apart, got {time_diff}s"


def test_normal_pattern_detection_algorithm_specifics():
    """Test specific behavior of normal pattern detection algorithm

    The normal pattern detection algorithm uses:
    - Partition-based MSE comparison (10 partitions, checks middle 2)
    - Pearson correlation of middle region for shape verification
    - Similarity threshold (<0.01)
    - Pearson r threshold (>=0.85)

    This test verifies the algorithm correctly identifies normal audio patterns.
    """
    peak_times, total_time = match_pattern(CBS_NEWS_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)

    matches = peak_times['cbs_news']

    # Should find exactly one match
    assert len(matches) == 1, "Normal pattern algorithm should find exactly 1 match"

    # Match should be in the latter part of the audio
    assert matches[0] > 20, \
        f"CBS news pattern should be found after 20s, got {matches[0]}s"


def test_correlation_peak_finding():
    """Test that correlation peaks are correctly identified

    The algorithm uses find_peaks with:
    - height_min = 0.25
    - distance = clip_length (no repetition within clip duration)

    This test verifies peak finding works correctly.
    """
    peak_times, total_time = match_pattern(RTHK_BEEP_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

    matches = peak_times['rthk_beep']

    # Should find distinct peaks
    assert len(matches) > 0, "Should find at least one peak"

    # All timestamps should be non-negative
    for match in matches:
        assert match >= 0, f"Timestamp should be non-negative, got {match}"


def test_loudness_normalization_effect():
    """Test that loudness normalization is applied correctly

    The algorithm normalizes both pattern and audio to -16 dB LUFS
    using pyloudnorm. This test verifies the normalization doesn't
    prevent pattern detection.
    """
    # Run pattern matching (normalization is enabled by default)
    peak_times, total_time = match_pattern(CBS_NEWS_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)

    # Should still find the pattern despite normalization
    assert 'cbs_news' in peak_times
    assert len(peak_times['cbs_news']) > 0, \
        "Loudness normalization should not prevent pattern detection"


# --- No Matching Patterns Tests ---


def test_beep_pattern_in_normal_audio():
    """Test that beep pattern does not match in CBS news audio

    RTHK beep is a pure tone pattern that should not match
    the complex CBS news audio patterns.
    """
    assert Path(RTHK_BEEP_PATTERN).exists(), f"Pattern file {RTHK_BEEP_PATTERN} not found"
    assert Path(CBS_NEWS_AUDIO).exists(), f"Audio file {CBS_NEWS_AUDIO} not found"

    peak_times, total_time = match_pattern(CBS_NEWS_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

    assert 'rthk_beep' in peak_times, "rthk_beep key should exist in results"
    assert len(peak_times['rthk_beep']) == 0, \
        f"RTHK beep should not match CBS news audio, but found {len(peak_times['rthk_beep'])} matches: {peak_times['rthk_beep']}"


def test_cbs_pattern_in_rthk_audio():
    """Test that CBS news pattern does not match in RTHK beep audio

    CBS news is a complex audio pattern that should not match
    the simple RTHK beep audio.
    """
    assert Path(CBS_NEWS_PATTERN).exists(), f"Pattern file {CBS_NEWS_PATTERN} not found"
    assert Path(RTHK_BEEP_AUDIO).exists(), f"Audio file {RTHK_BEEP_AUDIO} not found"

    peak_times, total_time = match_pattern(RTHK_BEEP_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)

    assert 'cbs_news' in peak_times, "cbs_news key should exist in results"
    assert len(peak_times['cbs_news']) == 0, \
        f"CBS news should not match RTHK audio, but found {len(peak_times['cbs_news'])} matches: {peak_times['cbs_news']}"


def test_multiple_patterns_none_match():
    """Test multiple patterns where none should match

    Tests that when multiple patterns are provided and none match,
    all patterns return empty results.
    """
    pattern_files = [CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN]
    audio_file = RTHK_BEEP_AUDIO

    for pattern_file in pattern_files:
        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
    assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

    peak_times, total_time = match_pattern(audio_file, pattern_files, debug_mode=False)

    # Both patterns should exist in results
    assert 'cbs_news' in peak_times, "cbs_news key should exist"
    assert '天空下的彩虹intro' in peak_times, "天空下的彩虹intro key should exist"

    # Both should have no matches
    assert len(peak_times['cbs_news']) == 0, \
        f"CBS news should not match, found {len(peak_times['cbs_news'])} matches"
    assert len(peak_times['天空下的彩虹intro']) == 0, \
        f"Rainbow intro should not match, found {len(peak_times['天空下的彩虹intro'])} matches"


def test_all_available_patterns_mixed_results():
    """Test all available patterns against all three audio files

    This comprehensive test verifies that each pattern only matches
    its own audio and produces no false positives on the others.
    """
    all_patterns = [RTHK_BEEP_PATTERN, CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN]

    # Test with RTHK audio
    assert Path(RTHK_BEEP_AUDIO).exists()
    rthk_results, _ = match_pattern(RTHK_BEEP_AUDIO, all_patterns, debug_mode=False)

    assert len(rthk_results['rthk_beep']) == 2, "RTHK beep should match in RTHK audio"
    assert len(rthk_results['cbs_news']) == 0, "CBS news should not match in RTHK audio"
    assert len(rthk_results['天空下的彩虹intro']) == 0, "Rainbow intro should not match in RTHK audio"

    # Test with CBS audio
    assert Path(CBS_NEWS_AUDIO).exists()
    cbs_results, _ = match_pattern(CBS_NEWS_AUDIO, all_patterns, debug_mode=False)

    assert len(cbs_results['cbs_news']) == 1, "CBS news should match in CBS audio"
    assert len(cbs_results['rthk_beep']) == 0, "RTHK beep should not match in CBS audio"
    assert len(cbs_results['天空下的彩虹intro']) == 0, "Rainbow intro should not match in CBS audio"

    # Test with AM1430 audio
    assert Path(RAINBOW_INTRO_AUDIO).exists()
    am_results, _ = match_pattern(RAINBOW_INTRO_AUDIO, all_patterns, debug_mode=False)

    assert len(am_results['天空下的彩虹intro']) == 1, "Rainbow intro should match in AM1430 audio"
    assert len(am_results['cbs_news']) == 0, "CBS news should not match in AM1430 audio"
    assert len(am_results['rthk_beep']) == 0, "RTHK beep should not match in AM1430 audio"


def test_similarity_threshold_rejection():
    """Test that patterns with similarity above threshold are rejected

    The normal pattern algorithm rejects matches with similarity > 0.01.
    This tests that dissimilar patterns are correctly filtered out.
    """
    # Using CBS news pattern on RTHK audio should produce high similarity
    # scores that exceed the threshold, resulting in rejection
    peak_times, _ = match_pattern(RTHK_BEEP_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)

    # Should be rejected due to high similarity scores
    assert len(peak_times['cbs_news']) == 0, \
        "Pattern should be rejected due to similarity threshold"


def test_overlap_ratio_rejection_for_beep():
    """Test that beep patterns with low overlap ratio are rejected

    The beep detection algorithm requires overlap_ratio > 0.98 or 0.99.
    This tests that patterns with insufficient overlap are filtered out.
    """
    # Using beep pattern on CBS audio should produce low overlap ratios
    peak_times, _ = match_pattern(CBS_NEWS_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

    # Should be rejected due to low overlap ratio
    assert len(peak_times['rthk_beep']) == 0, \
        "Beep pattern should be rejected due to low overlap ratio"


def test_no_false_positives_in_complex_scenario():
    """Test that complex cross-matching scenario produces no false positives

    This test verifies the robustness of both algorithms by testing
    all combinations of patterns and audio files where matches shouldn't occur.
    """
    test_cases = [
        # (pattern_file, audio_file, pattern_name)
        (RTHK_BEEP_PATTERN, CBS_NEWS_AUDIO, "rthk_beep"),
        (CBS_NEWS_PATTERN, RTHK_BEEP_AUDIO, "cbs_news"),
        (RAINBOW_INTRO_PATTERN, CBS_NEWS_AUDIO, "天空下的彩虹intro"),
        (RAINBOW_INTRO_PATTERN, RTHK_BEEP_AUDIO, "天空下的彩虹intro"),
    ]

    for pattern_file, audio_file, pattern_name in test_cases:
        assert Path(pattern_file).exists(), f"Pattern {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio {audio_file} not found"

        peak_times, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        assert pattern_name in peak_times, f"{pattern_name} key missing"
        assert len(peak_times[pattern_name]) == 0, \
            f"False positive detected: {pattern_name} in {Path(audio_file).name} " \
            f"produced {len(peak_times[pattern_name])} matches: {peak_times[pattern_name]}"


def test_correlation_peak_height_threshold():
    """Test that peaks below height_min=0.25 are filtered out

    The algorithm only considers correlation peaks with height >= 0.25.
    This test verifies that weak correlations don't produce matches.
    """
    # Mismatched patterns should produce low correlation peaks
    peak_times, _ = match_pattern(RTHK_BEEP_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)

    # Should have no matches due to low correlation peaks
    assert len(peak_times['cbs_news']) == 0, \
        "Low correlation peaks should not produce matches"


def test_verification_stage_filters_false_positives():
    """Test that the verification stage correctly filters false positives

    Even if correlation peaks are found, the verification stage should
    reject them based on:
    - MSE similarity threshold
    - Area overlap ratio
    - Diff overlap ratio

    This is a critical test for algorithm robustness.
    """
    # Test with patterns that might produce correlation peaks but should
    # be rejected in verification
    peak_times, _ = match_pattern(CBS_NEWS_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

    # Verification stage should reject any potential matches
    assert len(peak_times['rthk_beep']) == 0, \
        "Verification stage should filter out false positives"


# --- AM1430 Rainbow Intro (Lossy-Encoded Audio) Tests ---


def test_rainbow_intro_pattern_detection():
    """Test detection of 天空下的彩虹intro pattern in lossy-encoded audio

    This tests the normal pattern detection algorithm with audio that has
    been through Opus encoding, which degrades the cross-correlation shape.
    The Pearson correlation verification handles this by comparing
    downsampled envelope shapes rather than raw sample-by-sample values.

    Expected to find match at approximately 15.5s
    """
    assert Path(RAINBOW_INTRO_PATTERN).exists(), f"Pattern file {RAINBOW_INTRO_PATTERN} not found"
    assert Path(RAINBOW_INTRO_AUDIO).exists(), f"Audio file {RAINBOW_INTRO_AUDIO} not found"

    peak_times, total_time = match_pattern(RAINBOW_INTRO_AUDIO, [RAINBOW_INTRO_PATTERN], debug_mode=False)

    assert isinstance(peak_times, dict), "peak_times should be a dictionary"
    assert '天空下的彩虹intro' in peak_times, "天空下的彩虹intro pattern not found in results"

    matches = peak_times['天空下的彩虹intro']

    assert len(matches) == 1, f"Expected 1 match, found {len(matches)}: {matches}"

    actual_time = matches[0]
    assert abs(actual_time - RAINBOW_INTRO_EXPECTED_TIME) < 1.0, \
        f"Expected timestamp ~{RAINBOW_INTRO_EXPECTED_TIME}s, got {actual_time}s"

    assert total_time > 0, "Total processing time should be positive"


def test_rainbow_intro_not_in_cbs_audio():
    """Test that 天空下的彩虹intro pattern does not match CBS audio"""
    assert Path(RAINBOW_INTRO_PATTERN).exists(), f"Pattern file {RAINBOW_INTRO_PATTERN} not found"
    assert Path(CBS_NEWS_AUDIO).exists(), f"Audio file {CBS_NEWS_AUDIO} not found"

    peak_times, _ = match_pattern(CBS_NEWS_AUDIO, [RAINBOW_INTRO_PATTERN], debug_mode=False)

    assert '天空下的彩虹intro' in peak_times, "天空下的彩虹intro key should exist in results"
    assert len(peak_times['天空下的彩虹intro']) == 0, \
        f"天空下的彩虹intro should not match CBS audio, but found matches: {peak_times['天空下的彩虹intro']}"


def test_cbs_pattern_not_in_rainbow_intro_audio():
    """Test that CBS and RTHK patterns do not match in AM1430 rainbow intro audio"""
    test_cases = [
        (CBS_NEWS_PATTERN, "cbs_news"),
        (RTHK_BEEP_PATTERN, "rthk_beep"),
    ]
    audio_file = RAINBOW_INTRO_AUDIO

    assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

    for pattern_file, pattern_name in test_cases:
        assert Path(pattern_file).exists(), f"Pattern {pattern_file} not found"

        peak_times, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        assert pattern_name in peak_times, f"{pattern_name} key missing"
        assert len(peak_times[pattern_name]) == 0, \
            f"False positive: {pattern_name} in AM1430 audio produced {len(peak_times[pattern_name])} matches: {peak_times[pattern_name]}"


# --- 16kHz Audio Handling Tests ---


def test_match_16khz_audio_with_8khz_pattern():
    """Test matching 16kHz audio file against 8kHz pattern

    The match_pattern function should automatically convert 16kHz audio
    to 8kHz during processing and correctly identify patterns.
    """
    # Use 16kHz audio file
    audio_file = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"

    assert Path(RTHK_BEEP_PATTERN).exists(), f"Pattern file {RTHK_BEEP_PATTERN} not found"
    assert Path(audio_file).exists(), f"16kHz audio file {audio_file} not found"

    # Run pattern matching - should handle conversion automatically
    peak_times, total_time = match_pattern(audio_file, [RTHK_BEEP_PATTERN], debug_mode=False)

    # Should find the same matches as with 8kHz audio
    assert 'rthk_beep' in peak_times
    assert len(peak_times['rthk_beep']) == 2, \
        f"Expected 2 matches in 16kHz audio, found {len(peak_times['rthk_beep'])}"

    # Verify timestamps are similar (allowing small variation due to resampling)
    for i, (actual, expected) in enumerate(zip(sorted(peak_times['rthk_beep']), RTHK_BEEP_EXPECTED_TIMES)):
        assert abs(actual - expected) < 0.05, \
            f"Match {i}: Expected ~{expected}s, got {actual}s (tolerance increased for resampling)"


def test_match_16khz_cbs_news():
    """Test matching 16kHz CBS news audio against 8kHz pattern

    Tests normal pattern detection with 16kHz audio.
    """
    audio_file = "sample_audios/test_16khz/cbs_news_audio_section_16k.wav"

    assert Path(CBS_NEWS_PATTERN).exists(), f"Pattern file {CBS_NEWS_PATTERN} not found"
    assert Path(audio_file).exists(), f"16kHz audio file {audio_file} not found"

    peak_times, total_time = match_pattern(audio_file, [CBS_NEWS_PATTERN], debug_mode=False)

    assert 'cbs_news' in peak_times
    assert len(peak_times['cbs_news']) == 1, \
        f"Expected 1 match in 16kHz audio, found {len(peak_times['cbs_news'])}"

    # Verify timestamp (with increased tolerance for resampling)
    actual_time = peak_times['cbs_news'][0]
    assert abs(actual_time - CBS_NEWS_EXPECTED_TIME) < 0.05, \
        f"Expected ~{CBS_NEWS_EXPECTED_TIME}s, got {actual_time}s"


def test_match_16khz_with_converted_16khz_pattern():
    """Test matching 16kHz audio with pattern converted from 16kHz

    This tests the full workflow:
    1. Convert 16kHz pattern to 8kHz
    2. Match 16kHz audio against converted pattern
    3. Verify accuracy is maintained
    """
    # Convert 16kHz pattern to 8kHz
    input_pattern = "sample_audios/test_16khz/clips/rthk_beep_16k.wav"

    assert Path(input_pattern).exists(), f"16kHz pattern {input_pattern} not found"

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        converted_pattern = tmp.name

    try:
        # Step 1: Load 16kHz pattern and write as 8kHz
        audio = load_wave_file(input_pattern, 8000)
        write_wav_file(converted_pattern, audio, 8000)

        # Step 2: Match 16kHz audio against converted pattern
        audio_file = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"
        assert Path(audio_file).exists()

        peak_times, _ = match_pattern(audio_file, [converted_pattern], debug_mode=False)

        # Step 3: Verify results
        # Pattern name will be the temp file stem
        pattern_name = Path(converted_pattern).stem
        assert pattern_name in peak_times

        assert len(peak_times[pattern_name]) == 2, \
            f"Expected 2 matches, found {len(peak_times[pattern_name])}"

    finally:
        if Path(converted_pattern).exists():
            os.unlink(converted_pattern)


def test_multiple_16khz_patterns():
    """Test matching with multiple 16kHz-sourced patterns

    Tests that multiple patterns can be used simultaneously
    even when source files were originally 16kHz.
    """
    # Convert multiple 16kHz patterns to 8kHz
    input_patterns = [
        "sample_audios/test_16khz/clips/cbs_news_16k.wav",
        "sample_audios/test_16khz/clips/rthk_beep_16k.wav"
    ]

    converted_patterns = []
    temp_files = []

    try:
        # Convert all patterns
        for input_file in input_patterns:
            assert Path(input_file).exists(), f"Input pattern {input_file} not found"

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                output_file = tmp.name
                temp_files.append(output_file)

            audio = load_wave_file(input_file, 8000)
            write_wav_file(output_file, audio, 8000)
            converted_patterns.append(output_file)

        # Match against 16kHz CBS audio
        audio_file = "sample_audios/test_16khz/cbs_news_audio_section_16k.wav"
        assert Path(audio_file).exists()

        peak_times, _ = match_pattern(audio_file, converted_patterns, debug_mode=False)

        # Verify both patterns have result entries
        assert len(peak_times) == 2, "Expected 2 pattern results"

        # CBS news should match (1 match), rthk_beep should not match (0 matches)
        pattern_match_counts = sorted(len(matches) for matches in peak_times.values())
        assert pattern_match_counts == [0, 1], \
            f"Expected one pattern with 1 match and one with 0, got {dict(peak_times)}"

    finally:
        # Clean up all temp files
        for temp_file in temp_files:
            if Path(temp_file).exists():
                os.unlink(temp_file)


def test_16khz_no_false_positives():
    """Test that 16kHz audio doesn't produce false positives

    Verifies that sample rate conversion doesn't introduce
    false positive matches.
    """
    # Use CBS pattern with RTHK audio (should not match)
    audio_file = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"

    assert Path(CBS_NEWS_PATTERN).exists()
    assert Path(audio_file).exists()

    peak_times, _ = match_pattern(audio_file, [CBS_NEWS_PATTERN], debug_mode=False)

    assert 'cbs_news' in peak_times
    assert len(peak_times['cbs_news']) == 0, \
        f"16kHz conversion should not introduce false positives, found {len(peak_times['cbs_news'])} matches"


def test_16khz_beep_pattern_rejection():
    """Test that beep patterns correctly reject mismatches in 16kHz audio

    Verifies beep detection algorithm works correctly after
    sample rate conversion.
    """
    # Use RTHK beep with CBS audio (should not match)
    audio_file = "sample_audios/test_16khz/cbs_news_audio_section_16k.wav"

    assert Path(RTHK_BEEP_PATTERN).exists()
    assert Path(audio_file).exists()

    peak_times, _ = match_pattern(audio_file, [RTHK_BEEP_PATTERN], debug_mode=False)

    assert 'rthk_beep' in peak_times
    assert len(peak_times['rthk_beep']) == 0, \
        "Beep algorithm should reject mismatches in 16kHz audio"


def test_sample_rate_preservation_in_results():
    """Test that timestamps are correctly adjusted for sample rate conversion

    When 16kHz audio is converted to 8kHz, timestamps should still
    reflect the original audio timeline, not the converted timeline.
    """
    # Test with both 8kHz and 16kHz versions of the same audio
    audio_16k = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"

    assert Path(RTHK_BEEP_PATTERN).exists()
    assert Path(RTHK_BEEP_AUDIO).exists()
    assert Path(audio_16k).exists()

    # Match against 8kHz audio
    results_8k, _ = match_pattern(RTHK_BEEP_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

    # Match against 16kHz audio
    results_16k, _ = match_pattern(audio_16k, [RTHK_BEEP_PATTERN], debug_mode=False)

    # Both should find the same number of matches
    assert len(results_8k['rthk_beep']) == len(results_16k['rthk_beep']), \
        "Different sample rates should find same number of matches"

    # Timestamps should be similar (within tolerance for resampling)
    for i, (time_8k, time_16k) in enumerate(zip(
        sorted(results_8k['rthk_beep']),
        sorted(results_16k['rthk_beep'])
    )):
        assert abs(time_8k - time_16k) < 0.1, \
            f"Match {i}: Timestamps differ too much: 8kHz={time_8k}s, 16kHz={time_16k}s"


# --- Streaming Audio Processing Tests ---


def test_streaming_rthk_beep_detection():
    """Test streaming detection of RTHK beep pattern

    Uses AudioStream and AudioPatternDetector directly to test
    the streaming chunk-based processing.
    """
    assert Path(RTHK_BEEP_PATTERN).exists()
    assert Path(RTHK_BEEP_AUDIO).exists()

    # Load pattern clip
    pattern_clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)

    # Process audio using streaming
    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(RTHK_BEEP_AUDIO, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(RTHK_BEEP_AUDIO).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        # Run detection
        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    # Verify results
    assert 'rthk_beep' in peak_times
    assert len(peak_times['rthk_beep']) == 2, \
        f"Expected 2 matches, found {len(peak_times['rthk_beep'])}"

    for i, (actual, expected) in enumerate(zip(sorted(peak_times['rthk_beep']), RTHK_BEEP_EXPECTED_TIMES)):
        assert abs(actual - expected) < 0.01, \
            f"Match {i}: Expected ~{expected}s, got {actual}s"


def test_streaming_cbs_news_detection():
    """Test streaming detection of CBS news pattern

    Tests normal pattern detection through the streaming interface.
    """
    assert Path(CBS_NEWS_PATTERN).exists()
    assert Path(CBS_NEWS_AUDIO).exists()

    pattern_clip = AudioClip.from_audio_file(CBS_NEWS_PATTERN)

    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(CBS_NEWS_AUDIO, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(CBS_NEWS_AUDIO).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    assert 'cbs_news' in peak_times
    assert len(peak_times['cbs_news']) == 1

    assert abs(peak_times['cbs_news'][0] - CBS_NEWS_EXPECTED_TIME) < 0.01


def test_streaming_multiple_patterns():
    """Test streaming detection with multiple patterns simultaneously

    Verifies that multiple AudioClips can be processed in a single stream.
    Tests CBS news and rainbow intro patterns against CBS audio;
    CBS news should match while rainbow intro should not.
    """
    pattern_files = [CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN]
    audio_file = CBS_NEWS_AUDIO

    assert Path(audio_file).exists()

    # Load multiple pattern clips
    pattern_clips = []
    for pf in pattern_files:
        assert Path(pf).exists()
        pattern_clips.append(AudioClip.from_audio_file(pf))

    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=pattern_clips)
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    # CBS news should match, rainbow intro should not
    assert 'cbs_news' in peak_times
    assert '天空下的彩虹intro' in peak_times
    assert len(peak_times['cbs_news']) == 1
    assert len(peak_times['天空下的彩虹intro']) == 0


def test_streaming_16khz_audio_conversion():
    """Test streaming with 16kHz audio auto-conversion

    Verifies that ffmpeg_get_float32_pcm correctly converts 16kHz to 8kHz
    during streaming and pattern detection works correctly.
    """
    audio_file = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"

    assert Path(RTHK_BEEP_PATTERN).exists()
    assert Path(audio_file).exists()

    pattern_clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)

    # Stream 16kHz audio with conversion to 8kHz
    sr = DEFAULT_TARGET_SAMPLE_RATE  # 8000 Hz
    with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    # Should find same matches as with native 8kHz audio
    assert 'rthk_beep' in peak_times
    assert len(peak_times['rthk_beep']) == 2

    for i, (actual, expected) in enumerate(zip(sorted(peak_times['rthk_beep']), RTHK_BEEP_EXPECTED_TIMES)):
        assert abs(actual - expected) < 0.05, \
            f"Match {i}: Expected ~{expected}s, got {actual}s"


def test_streaming_chunk_processing():
    """Test that streaming processes audio in chunks correctly

    Verifies the chunked processing maintains accuracy when
    pattern spans chunk boundaries.
    """
    pattern_clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)

    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(RTHK_BEEP_AUDIO, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(RTHK_BEEP_AUDIO).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        # Use default seconds_per_chunk (60 seconds)
        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern_clip],
            seconds_per_chunk=60
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    # Should still find all matches regardless of chunking
    assert len(peak_times['rthk_beep']) == 2


def test_streaming_small_chunk_size():
    """Test streaming with smaller chunk size

    Tests that the detector handles smaller chunks correctly.
    Note: With small chunk sizes and sliding window overlap,
    the same pattern may be detected in multiple chunks,
    resulting in duplicate timestamps. This tests that the
    correct timestamps ARE found (duplicates may exist).
    """
    pattern_clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)

    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(RTHK_BEEP_AUDIO, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(RTHK_BEEP_AUDIO).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        # Use smaller chunk size (must be at least 2x sliding_window)
        # rthk_beep is ~0.23s, so sliding_window rounds to 1s
        # Minimum chunk size is 2s
        detector = AudioPatternDetector(
            debug_mode=False,
            audio_clips=[pattern_clip],
            seconds_per_chunk=3
        )
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    assert 'rthk_beep' in peak_times

    # With small chunks and sliding window overlap, duplicates may occur
    # Verify at least 2 matches found
    assert len(peak_times['rthk_beep']) >= 2

    # Verify expected timestamps are present (use set to handle duplicates)
    found_times: set[float] = set()
    for actual in peak_times['rthk_beep']:
        for expected in RTHK_BEEP_EXPECTED_TIMES:
            if abs(actual - expected) < 0.01:
                found_times.add(expected)
                break

    assert len(found_times) == len(RTHK_BEEP_EXPECTED_TIMES), \
        f"Expected to find timestamps near {RTHK_BEEP_EXPECTED_TIMES}, found {peak_times['rthk_beep']}"


def test_streaming_no_match_scenario():
    """Test streaming when pattern is not present in audio

    Verifies that streaming processing correctly returns empty results.
    """
    pattern_clip = AudioClip.from_audio_file(CBS_NEWS_PATTERN)

    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(RTHK_BEEP_AUDIO, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(RTHK_BEEP_AUDIO).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    assert 'cbs_news' in peak_times
    assert len(peak_times['cbs_news']) == 0


def test_streaming_total_time_accuracy():
    """Test that total_time accurately reflects processed audio duration

    Verifies the streaming processor correctly tracks total audio processed.
    """
    pattern_clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)

    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(RTHK_BEEP_AUDIO, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(RTHK_BEEP_AUDIO).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    # rthk_section_with_beep.wav is ~4.08 seconds
    assert 4.0 < total_time < 4.2, f"Expected ~4.08s, got {total_time}s"


def test_audio_clip_from_file():
    """Test AudioClip.from_audio_file correctly loads pattern files

    Verifies AudioClip dataclass is properly initialized.
    """
    clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)

    assert clip.name == "rthk_beep"
    assert clip.sample_rate == DEFAULT_TARGET_SAMPLE_RATE
    assert len(clip.audio) > 0
    assert clip.clip_length_seconds() > 0


def test_audio_clip_sample_rate_validation():
    """Test that AudioPatternDetector validates pattern sample rates

    Patterns must match DEFAULT_TARGET_SAMPLE_RATE (8000 Hz).
    """
    # Create clip with correct sample rate
    pattern_clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)

    # Verify clip has correct sample rate
    assert pattern_clip.sample_rate == DEFAULT_TARGET_SAMPLE_RATE

    # Detector should accept valid clips
    detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
    assert len(detector.audio_clips) == 1


def test_streaming_maintains_pattern_order():
    """Test that multiple patterns maintain their result order

    Verifies each pattern's results are stored under the correct key.
    """
    pattern_files = [RTHK_BEEP_PATTERN, CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN]
    audio_file = CBS_NEWS_AUDIO

    pattern_clips = [AudioClip.from_audio_file(pf) for pf in pattern_files]

    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=pattern_clips)
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    # All patterns should have result entries
    assert 'rthk_beep' in peak_times
    assert 'cbs_news' in peak_times
    assert '天空下的彩虹intro' in peak_times

    # RTHK beep shouldn't match CBS audio
    assert len(peak_times['rthk_beep']) == 0
    # CBS news should match
    assert len(peak_times['cbs_news']) == 1
    # Rainbow intro shouldn't match CBS audio
    assert len(peak_times['天空下的彩虹intro']) == 0


def test_streaming_duplicate_pattern_names_rejected():
    """Test that duplicate pattern names are rejected

    AudioPatternDetector should raise ValueError for duplicate clip names.
    """
    clip1 = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)
    clip2 = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)  # Same name

    with pytest.raises(ValueError, match="needs to be unique"):
        AudioPatternDetector(debug_mode=False, audio_clips=[clip1, clip2])


def test_streaming_16khz_cbs_news():
    """Test streaming 16kHz CBS news audio with conversion

    Tests normal pattern detection through streaming with sample rate conversion.
    """
    audio_file = "sample_audios/test_16khz/cbs_news_audio_section_16k.wav"

    assert Path(CBS_NEWS_PATTERN).exists()
    assert Path(audio_file).exists()

    pattern_clip = AudioClip.from_audio_file(CBS_NEWS_PATTERN)

    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        peak_times, total_time = detector.find_clip_in_audio(audio_stream)

    assert 'cbs_news' in peak_times
    assert len(peak_times['cbs_news']) == 1

    assert abs(peak_times['cbs_news'][0] - CBS_NEWS_EXPECTED_TIME) < 0.05


def test_streaming_results_match_high_level_api():
    """Test that streaming results match the high-level match_pattern API

    Ensures consistency between low-level streaming and high-level APIs.
    """
    # High-level API result
    high_level_results, _ = match_pattern(RTHK_BEEP_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

    # Low-level streaming result
    pattern_clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)
    sr = DEFAULT_TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(RTHK_BEEP_AUDIO, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(RTHK_BEEP_AUDIO).stem
        audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        streaming_results, _ = detector.find_clip_in_audio(audio_stream)

    # Results should be identical
    assert len(high_level_results['rthk_beep']) == len(streaming_results['rthk_beep'])

    for hl, st in zip(
        sorted(high_level_results['rthk_beep']),
        sorted(streaming_results['rthk_beep'])
    ):
        assert abs(hl - st) < 0.001, f"Results differ: high-level={hl}, streaming={st}"


# --- WAV File Processing Without FFmpeg Tests ---


class TestWavFileStreamWrapper:
    """Tests for _WavFileStreamWrapper class (ffmpeg-free WAV file streaming)."""

    def test_wav_file_stream_wrapper_basic(self):
        """Test basic functionality of _WavFileStreamWrapper."""
        assert Path(RTHK_BEEP_PATTERN).exists()

        wrapper = _WavFileStreamWrapper(RTHK_BEEP_PATTERN, DEFAULT_TARGET_SAMPLE_RATE)
        try:
            assert wrapper.target_sample_rate == DEFAULT_TARGET_SAMPLE_RATE
            assert wrapper.input_sample_rate > 0
            assert wrapper._channels >= 1
            assert wrapper._sampwidth > 0
        finally:
            wrapper.close()

    def test_wav_file_stream_wrapper_read(self):
        """Test reading data from _WavFileStreamWrapper."""
        import numpy as np

        assert Path(RTHK_BEEP_PATTERN).exists()

        wrapper = _WavFileStreamWrapper(RTHK_BEEP_PATTERN, DEFAULT_TARGET_SAMPLE_RATE)
        try:
            # Read some data (4 bytes per float32 sample)
            data = wrapper.read(4000)  # 1000 samples
            assert len(data) > 0

            # Convert to numpy and verify
            audio = np.frombuffer(data, dtype=np.float32)
            assert len(audio) > 0
            assert audio.dtype == np.float32
            # Audio should be normalized
            assert np.max(np.abs(audio)) <= 1.5
        finally:
            wrapper.close()

    def test_wav_file_stream_wrapper_full_read(self):
        """Test reading entire WAV file via _WavFileStreamWrapper."""
        import numpy as np

        assert Path(RTHK_BEEP_PATTERN).exists()

        wrapper = _WavFileStreamWrapper(RTHK_BEEP_PATTERN, DEFAULT_TARGET_SAMPLE_RATE)
        try:
            # Read entire file in chunks
            all_data = b""
            while True:
                chunk = wrapper.read(32000)  # 8000 samples per read
                if not chunk:
                    break
                all_data += chunk

            audio = np.frombuffer(all_data, dtype=np.float32)
            assert len(audio) > 0
            # rthk_beep is ~0.23 seconds at 8kHz = ~1840 samples
            assert 1500 < len(audio) < 2500
        finally:
            wrapper.close()

    def test_wav_file_stream_wrapper_resampling(self):
        """Test that _WavFileStreamWrapper correctly resamples audio."""
        import numpy as np

        # Use 16kHz file to test resampling to 8kHz
        wav_file = "sample_audios/test_16khz/clips/rthk_beep_16k.wav"
        if not Path(wav_file).exists():
            pytest.skip("16kHz test file not found")

        wrapper = _WavFileStreamWrapper(wav_file, 8000)
        try:
            assert wrapper.input_sample_rate == 16000
            assert wrapper.target_sample_rate == 8000
            assert wrapper.needs_resample is True

            # Read entire file
            all_data = b""
            while True:
                chunk = wrapper.read(32000)
                if not chunk:
                    break
                all_data += chunk

            audio = np.frombuffer(all_data, dtype=np.float32)
            # Should be at 8kHz, not 16kHz
            # Duration should be preserved (~0.23s = ~1840 samples at 8kHz)
            assert 1500 < len(audio) < 2500
        finally:
            wrapper.close()

    def test_wav_file_stream_wrapper_no_resampling(self):
        """Test _WavFileStreamWrapper when no resampling is needed."""
        assert Path(RTHK_BEEP_PATTERN).exists()

        wrapper = _WavFileStreamWrapper(RTHK_BEEP_PATTERN, 8000)  # Already 8kHz
        try:
            assert wrapper.input_sample_rate == 8000
            assert wrapper.target_sample_rate == 8000
            assert wrapper.needs_resample is False
        finally:
            wrapper.close()

    def test_wav_file_stream_wrapper_nonexistent_file(self):
        """Test that _WavFileStreamWrapper raises error for nonexistent file."""
        with pytest.raises(ValueError, match="Failed to read WAV file"):
            _WavFileStreamWrapper("nonexistent.wav", 8000)

    def test_wav_file_stream_wrapper_with_audio_stream(self):
        """Test using _WavFileStreamWrapper with AudioStream class."""
        assert Path(RTHK_BEEP_PATTERN).exists()

        wrapper = _WavFileStreamWrapper(RTHK_BEEP_PATTERN, DEFAULT_TARGET_SAMPLE_RATE)
        try:
            audio_stream = AudioStream(
                name="test_stream",
                audio_stream=wrapper,
                sample_rate=DEFAULT_TARGET_SAMPLE_RATE
            )
            assert audio_stream.name == "test_stream"
            assert audio_stream.sample_rate == DEFAULT_TARGET_SAMPLE_RATE
        finally:
            wrapper.close()


class TestWavFileMatchingWithoutFfmpeg:
    """Tests for WAV file pattern matching without ffmpeg."""

    def test_wav_match_without_ffmpeg(self):
        """Test that WAV file matching works without ffmpeg."""
        assert Path(RTHK_BEEP_PATTERN).exists()
        assert Path(RTHK_BEEP_AUDIO).exists()

        # This should use _WavFileStreamWrapper internally
        peak_times, total_time = match_pattern(RTHK_BEEP_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

        # Should find the expected matches
        assert 'rthk_beep' in peak_times
        assert len(peak_times['rthk_beep']) == 2

        for i, (actual, expected) in enumerate(zip(sorted(peak_times['rthk_beep']), RTHK_BEEP_EXPECTED_TIMES)):
            assert abs(actual - expected) < 0.01

    def test_wav_match_results_consistent(self):
        """Test that WAV matching produces consistent results with streaming API."""
        assert Path(CBS_NEWS_PATTERN).exists()
        assert Path(CBS_NEWS_AUDIO).exists()

        # Run match_pattern
        peak_times, total_time = match_pattern(CBS_NEWS_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)

        assert 'cbs_news' in peak_times
        assert len(peak_times['cbs_news']) == 1

        assert abs(peak_times['cbs_news'][0] - CBS_NEWS_EXPECTED_TIME) < 0.01

    def test_wav_match_16khz_resampling(self):
        """Test WAV matching with 16kHz file (resampled to 8kHz)."""
        audio_file_16k = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"

        assert Path(RTHK_BEEP_PATTERN).exists()
        if not Path(audio_file_16k).exists():
            pytest.skip("16kHz test file not found")

        peak_times, total_time = match_pattern(audio_file_16k, [RTHK_BEEP_PATTERN], debug_mode=False)

        assert 'rthk_beep' in peak_times
        assert len(peak_times['rthk_beep']) == 2

        for i, (actual, expected) in enumerate(zip(sorted(peak_times['rthk_beep']), RTHK_BEEP_EXPECTED_TIMES)):
            assert abs(actual - expected) < 0.05

    def test_wav_match_no_false_positives(self):
        """Test that WAV matching doesn't produce false positives."""
        assert Path(CBS_NEWS_PATTERN).exists()
        assert Path(RTHK_BEEP_AUDIO).exists()

        peak_times, total_time = match_pattern(RTHK_BEEP_AUDIO, [CBS_NEWS_PATTERN], debug_mode=False)

        assert 'cbs_news' in peak_times
        assert len(peak_times['cbs_news']) == 0

    def test_wav_match_multiple_patterns(self):
        """Test WAV matching with multiple patterns."""
        pattern_files = [CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN]
        audio_file = CBS_NEWS_AUDIO

        for pf in pattern_files:
            assert Path(pf).exists()
        assert Path(audio_file).exists()

        peak_times, total_time = match_pattern(audio_file, pattern_files, debug_mode=False)

        assert 'cbs_news' in peak_times
        assert '天空下的彩虹intro' in peak_times
        assert len(peak_times['cbs_news']) == 1
        assert len(peak_times['天空下的彩虹intro']) == 0

    def test_wav_match_without_ffmpeg_available(self):
        """Test WAV file matching works when ffmpeg is not available."""
        from audio_pattern_detector import audio_utils

        assert Path(RTHK_BEEP_PATTERN).exists()
        assert Path(RTHK_BEEP_AUDIO).exists()

        # Mock ffmpeg as unavailable
        original_state = audio_utils._ffmpeg_available
        audio_utils._ffmpeg_available = False

        try:
            # Should still work for WAV files
            peak_times, total_time = match_pattern(RTHK_BEEP_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

            assert 'rthk_beep' in peak_times
            assert len(peak_times['rthk_beep']) == 2

            for i, (actual, expected) in enumerate(zip(sorted(peak_times['rthk_beep']), RTHK_BEEP_EXPECTED_TIMES)):
                assert abs(actual - expected) < 0.01
        finally:
            audio_utils._ffmpeg_available = original_state

    def test_wav_match_streaming_with_wrapper(self):
        """Test streaming pattern matching using _WavFileStreamWrapper directly."""
        assert Path(RTHK_BEEP_PATTERN).exists()
        assert Path(RTHK_BEEP_AUDIO).exists()

        # Load pattern
        pattern_clip = AudioClip.from_audio_file(RTHK_BEEP_PATTERN)

        # Use _WavFileStreamWrapper directly
        wrapper = _WavFileStreamWrapper(RTHK_BEEP_AUDIO, DEFAULT_TARGET_SAMPLE_RATE)
        try:
            audio_stream = AudioStream(
                name=Path(RTHK_BEEP_AUDIO).stem,
                audio_stream=wrapper,
                sample_rate=DEFAULT_TARGET_SAMPLE_RATE
            )

            detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
            peak_times, total_time = detector.find_clip_in_audio(audio_stream)

            assert 'rthk_beep' in peak_times
            assert len(peak_times['rthk_beep']) == 2
        finally:
            wrapper.close()

    def test_wav_match_total_time_accuracy(self):
        """Test that total_time is accurate for WAV file matching."""
        assert Path(RTHK_BEEP_PATTERN).exists()
        assert Path(RTHK_BEEP_AUDIO).exists()

        peak_times, total_time = match_pattern(RTHK_BEEP_AUDIO, [RTHK_BEEP_PATTERN], debug_mode=False)

        # rthk_section_with_beep.wav is ~4.08 seconds
        assert 4.0 < total_time < 4.2, f"Expected ~4.08s, got {total_time}s"

    def test_wav_match_stereo_file(self):
        """Test WAV matching with stereo file (converted to mono)."""
        # Create a temporary stereo WAV file for testing
        import subprocess
        import numpy as np

        sample_rate = 8000
        duration_seconds = 1
        num_samples = sample_rate * duration_seconds

        # Create stereo data
        audio_left = np.sin(2 * np.pi * 440 * np.arange(num_samples) / sample_rate)
        audio_right = np.sin(2 * np.pi * 880 * np.arange(num_samples) / sample_rate)
        stereo_int16 = (np.column_stack((audio_left, audio_right)) * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            stereo_file = f.name

        try:
            # Create stereo WAV using ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-f", "s16le",
                "-ar", str(sample_rate),
                "-ac", "2",
                "-i", "pipe:",
                "-loglevel", "error",
                stereo_file
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            proc.communicate(stereo_int16.tobytes())

            # Test that _WavFileStreamWrapper can handle stereo
            wrapper = _WavFileStreamWrapper(stereo_file, sample_rate)
            try:
                assert wrapper._channels == 2

                # Read data - should be converted to mono
                data = wrapper.read(4000)
                audio = np.frombuffer(data, dtype=np.float32)
                assert len(audio) > 0
            finally:
                wrapper.close()
        finally:
            os.unlink(stereo_file)
