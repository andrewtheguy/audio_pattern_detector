import os
from pathlib import Path

import pytest

from audio_pattern_detector.match import match_pattern


class TestPatternMatching:
    """Integration tests for audio pattern matching logic"""

    def test_rthk_beep_pattern_detection(self):
        """Test detection of RTHK beep pattern (pure tone/beep detection)

        This tests the beep detection algorithm (_get_peak_times_beep_v3)
        which uses overlap_ratio and downsampled correlation matching.

        Expected to find matches at approximately 1.4165s and 2.419125s
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        # Verify input files exist
        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        # Run pattern matching
        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Verify results structure
        assert isinstance(peak_times, dict), "peak_times should be a dictionary"
        assert 'rthk_beep' in peak_times, "rthk_beep pattern not found in results"

        matches = peak_times['rthk_beep']

        # Verify we found 2 matches
        assert len(matches) == 2, f"Expected 2 matches, found {len(matches)}: {matches}"

        # Verify the timestamps (with tolerance for floating point comparison)
        expected_times = [1.4165, 2.419125]
        for i, (actual, expected) in enumerate(zip(sorted(matches), expected_times)):
            assert abs(actual - expected) < 0.01, \
                f"Match {i}: Expected timestamp ~{expected}s, got {actual}s"

        # Verify processing time is reasonable
        assert total_time > 0, "Total processing time should be positive"
        assert total_time < 10, f"Processing took too long: {total_time}s"

    def test_cbs_news_pattern_detection(self):
        """Test detection of CBS News pattern (normal audio pattern)

        This tests the normal pattern detection algorithm (_get_peak_times_normal)
        which uses mean squared error and area overlap ratio.

        Expected to find match at approximately 25.89875s
        """
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        # Verify input files exist
        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        # Run pattern matching
        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Verify results structure
        assert isinstance(peak_times, dict), "peak_times should be a dictionary"
        assert 'cbs_news' in peak_times, "cbs_news pattern not found in results"

        matches = peak_times['cbs_news']

        # Verify we found 1 match
        assert len(matches) == 1, f"Expected 1 match, found {len(matches)}: {matches}"

        # Verify the timestamp (with tolerance for floating point comparison)
        expected_time = 25.89875
        actual_time = matches[0]
        assert abs(actual_time - expected_time) < 0.01, \
            f"Expected timestamp ~{expected_time}s, got {actual_time}s"

        # Verify processing time is reasonable
        assert total_time > 0, "Total processing time should be positive"

    def test_cbs_news_dada_pattern_detection(self):
        """Test detection of CBS News dada pattern (shorter normal pattern)

        This tests the normal pattern detection with a shorter clip.
        The algorithm should handle verification failures and still find the correct match.

        Expected to find match at approximately 1.965625s
        """
        pattern_file = "sample_audios/clips/cbs_news_dada.wav"
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        # Verify input files exist
        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        # Run pattern matching
        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Verify results structure
        assert isinstance(peak_times, dict), "peak_times should be a dictionary"
        assert 'cbs_news_dada' in peak_times, "cbs_news_dada pattern not found in results"

        matches = peak_times['cbs_news_dada']

        # Verify we found 1 match
        assert len(matches) == 1, f"Expected 1 match, found {len(matches)}: {matches}"

        # Verify the timestamp (with tolerance for floating point comparison)
        expected_time = 1.965625
        actual_time = matches[0]
        assert abs(actual_time - expected_time) < 0.01, \
            f"Expected timestamp ~{expected_time}s, got {actual_time}s"

        # Verify processing time is reasonable
        assert total_time > 0, "Total processing time should be positive"

    def test_multiple_patterns_single_audio(self):
        """Test matching multiple patterns in a single audio file

        Tests that the detector can handle multiple patterns simultaneously
        and correctly identify each pattern type.
        """
        pattern_files = [
            "sample_audios/clips/cbs_news.wav",
            "sample_audios/clips/cbs_news_dada.wav"
        ]
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        # Verify all files exist
        for pattern_file in pattern_files:
            assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        # Run pattern matching with multiple patterns
        peak_times, total_time = match_pattern(audio_file, pattern_files, debug_mode=False)

        # Verify both patterns were found
        assert 'cbs_news' in peak_times, "cbs_news pattern not found"
        assert 'cbs_news_dada' in peak_times, "cbs_news_dada pattern not found"

        # Verify cbs_news match
        assert len(peak_times['cbs_news']) == 1
        assert abs(peak_times['cbs_news'][0] - 25.89875) < 0.01

        # Verify cbs_news_dada match
        assert len(peak_times['cbs_news_dada']) == 1
        assert abs(peak_times['cbs_news_dada'][0] - 1.965625) < 0.01

    def test_pattern_not_in_audio(self):
        """Test that pattern matching returns empty results when pattern is not present

        Tests the algorithm's ability to correctly reject false positives.
        """
        # Use CBS news pattern on RTHK audio (should not match)
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        # Verify files exist
        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        # Run pattern matching
        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Should return empty results (no matches)
        assert 'cbs_news' in peak_times, "cbs_news key should exist in results"
        assert len(peak_times['cbs_news']) == 0, \
            f"Expected no matches, but found {len(peak_times['cbs_news'])}: {peak_times['cbs_news']}"

    def test_nonexistent_pattern_file(self):
        """Test that match_pattern raises ValueError for nonexistent pattern file"""
        pattern_file = "sample_audios/clips/nonexistent.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        with pytest.raises(ValueError, match="does not exist"):
            match_pattern(audio_file, [pattern_file], debug_mode=False)

    def test_nonexistent_audio_file(self):
        """Test that match_pattern raises ValueError for nonexistent audio file"""
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/nonexistent.wav"

        with pytest.raises(ValueError, match="does not exist"):
            match_pattern(audio_file, [pattern_file], debug_mode=False)

    def test_empty_pattern_list(self):
        """Test that match_pattern raises ValueError when no patterns provided"""
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        with pytest.raises(ValueError, match="No pattern clips passed"):
            match_pattern(audio_file, [], debug_mode=False)

    def test_debug_mode_execution(self):
        """Test that debug mode executes without errors

        Debug mode enables additional logging and graph generation.
        This test verifies it doesn't crash the detection process.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        # Verify files exist
        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        # Run with debug mode enabled
        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=True)

        # Should still find the correct matches
        assert 'rthk_beep' in peak_times
        assert len(peak_times['rthk_beep']) == 2

        # Debug mode should create graph directories (but we won't check file contents)
        # Just verify the function completed successfully

    def test_beep_detection_algorithm_specifics(self):
        """Test specific behavior of beep detection algorithm

        The beep detection algorithm uses:
        - Downsampled correlation matching
        - Overlap ratio threshold (>0.98 or >0.99 depending on similarity)
        - MSE similarity threshold (<0.01)

        This test verifies the algorithm correctly identifies pure tone patterns.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        matches = peak_times['rthk_beep']

        # The algorithm should find exactly 2 beeps
        assert len(matches) == 2, "Beep algorithm should find exactly 2 matches"

        # Matches should be in chronological order
        assert matches[0] < matches[1], "Matches should be sorted chronologically"

        # Matches should be separated by a reasonable time
        time_diff = matches[1] - matches[0]
        assert 0.5 < time_diff < 5.0, \
            f"Beeps should be 0.5-5s apart, got {time_diff}s"

    def test_normal_pattern_detection_algorithm_specifics(self):
        """Test specific behavior of normal pattern detection algorithm

        The normal pattern detection algorithm uses:
        - Partition-based MSE comparison (10 partitions, checks middle 2)
        - Area overlap ratio
        - Similarity threshold (<0.01)
        - Diff overlap ratio threshold (<0.5)

        This test verifies the algorithm correctly identifies normal audio patterns.
        """
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        matches = peak_times['cbs_news']

        # Should find exactly one match
        assert len(matches) == 1, "Normal pattern algorithm should find exactly 1 match"

        # Match should be in the latter part of the audio
        assert matches[0] > 20, \
            f"CBS news pattern should be found after 20s, got {matches[0]}s"

    def test_correlation_peak_finding(self):
        """Test that correlation peaks are correctly identified

        The algorithm uses scipy.signal.find_peaks with:
        - height_min = 0.25
        - distance = clip_length (no repetition within clip duration)

        This test verifies peak finding works correctly.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        matches = peak_times['rthk_beep']

        # Should find distinct peaks
        assert len(matches) > 0, "Should find at least one peak"

        # All timestamps should be non-negative
        for match in matches:
            assert match >= 0, f"Timestamp should be non-negative, got {match}"

    def test_loudness_normalization_effect(self):
        """Test that loudness normalization is applied correctly

        The algorithm normalizes both pattern and audio to -16 dB LUFS
        using pyloudnorm. This test verifies the normalization doesn't
        prevent pattern detection.
        """
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        # Run pattern matching (normalization is enabled by default)
        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Should still find the pattern despite normalization
        assert 'cbs_news' in peak_times
        assert len(peak_times['cbs_news']) > 0, \
            "Loudness normalization should not prevent pattern detection"