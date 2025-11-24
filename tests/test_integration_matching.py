import os
import tempfile
from pathlib import Path

import pytest

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import ffmpeg_get_16bit_pcm, TARGET_SAMPLE_RATE
from audio_pattern_detector.convert import convert_audio_to_clip_format
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


class TestNoMatchingPatterns:
    """Extended tests for scenarios where patterns should not match"""

    def test_beep_pattern_in_normal_audio(self):
        """Test that beep pattern does not match in CBS news audio

        RTHK beep is a pure tone pattern that should not match
        the complex CBS news audio patterns.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        assert 'rthk_beep' in peak_times, "rthk_beep key should exist in results"
        assert len(peak_times['rthk_beep']) == 0, \
            f"RTHK beep should not match CBS news audio, but found {len(peak_times['rthk_beep'])} matches: {peak_times['rthk_beep']}"

    def test_cbs_pattern_in_rthk_audio(self):
        """Test that CBS news pattern does not match in RTHK beep audio

        CBS news is a complex audio pattern that should not match
        the simple RTHK beep audio.
        """
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        assert 'cbs_news' in peak_times, "cbs_news key should exist in results"
        assert len(peak_times['cbs_news']) == 0, \
            f"CBS news should not match RTHK audio, but found {len(peak_times['cbs_news'])} matches: {peak_times['cbs_news']}"

    def test_dada_pattern_in_rthk_audio(self):
        """Test that CBS news dada pattern does not match in RTHK audio

        Tests that a shorter CBS pattern still doesn't match unrelated audio.
        """
        pattern_file = "sample_audios/clips/cbs_news_dada.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        assert 'cbs_news_dada' in peak_times, "cbs_news_dada key should exist in results"
        assert len(peak_times['cbs_news_dada']) == 0, \
            f"CBS news dada should not match RTHK audio, but found {len(peak_times['cbs_news_dada'])} matches: {peak_times['cbs_news_dada']}"

    def test_multiple_patterns_none_match(self):
        """Test multiple patterns where none should match

        Tests that when multiple patterns are provided and none match,
        all patterns return empty results.
        """
        pattern_files = [
            "sample_audios/clips/cbs_news.wav",
            "sample_audios/clips/cbs_news_dada.wav"
        ]
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        for pattern_file in pattern_files:
            assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

        peak_times, total_time = match_pattern(audio_file, pattern_files, debug_mode=False)

        # Both patterns should exist in results
        assert 'cbs_news' in peak_times, "cbs_news key should exist"
        assert 'cbs_news_dada' in peak_times, "cbs_news_dada key should exist"

        # Both should have no matches
        assert len(peak_times['cbs_news']) == 0, \
            f"CBS news should not match, found {len(peak_times['cbs_news'])} matches"
        assert len(peak_times['cbs_news_dada']) == 0, \
            f"CBS news dada should not match, found {len(peak_times['cbs_news_dada'])} matches"

    def test_all_available_patterns_mixed_results(self):
        """Test all available patterns against both audio files

        This comprehensive test verifies that:
        - RTHK beep matches in RTHK audio but not CBS audio
        - CBS patterns match in CBS audio but not RTHK audio
        """
        all_patterns = [
            "sample_audios/clips/rthk_beep.wav",
            "sample_audios/clips/cbs_news.wav",
            "sample_audios/clips/cbs_news_dada.wav"
        ]

        # Test with RTHK audio
        rthk_audio = "sample_audios/rthk_section_with_beep.wav"
        assert Path(rthk_audio).exists()

        rthk_results, _ = match_pattern(rthk_audio, all_patterns, debug_mode=False)

        # RTHK beep should match
        assert len(rthk_results['rthk_beep']) == 2, "RTHK beep should match in RTHK audio"
        # CBS patterns should not match
        assert len(rthk_results['cbs_news']) == 0, "CBS news should not match in RTHK audio"
        assert len(rthk_results['cbs_news_dada']) == 0, "CBS dada should not match in RTHK audio"

        # Test with CBS audio
        cbs_audio = "sample_audios/cbs_news_audio_section.wav"
        assert Path(cbs_audio).exists()

        cbs_results, _ = match_pattern(cbs_audio, all_patterns, debug_mode=False)

        # CBS patterns should match
        assert len(cbs_results['cbs_news']) == 1, "CBS news should match in CBS audio"
        assert len(cbs_results['cbs_news_dada']) == 1, "CBS dada should match in CBS audio"
        # RTHK beep should not match
        assert len(cbs_results['rthk_beep']) == 0, "RTHK beep should not match in CBS audio"

    def test_similarity_threshold_rejection(self):
        """Test that patterns with similarity above threshold are rejected

        The normal pattern algorithm rejects matches with similarity > 0.01.
        This tests that dissimilar patterns are correctly filtered out.
        """
        # Using CBS news pattern on RTHK audio should produce high similarity
        # scores that exceed the threshold, resulting in rejection
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        peak_times, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Should be rejected due to high similarity scores
        assert len(peak_times['cbs_news']) == 0, \
            "Pattern should be rejected due to similarity threshold"

    def test_overlap_ratio_rejection_for_beep(self):
        """Test that beep patterns with low overlap ratio are rejected

        The beep detection algorithm requires overlap_ratio > 0.98 or 0.99.
        This tests that patterns with insufficient overlap are filtered out.
        """
        # Using beep pattern on CBS audio should produce low overlap ratios
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        peak_times, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Should be rejected due to low overlap ratio
        assert len(peak_times['rthk_beep']) == 0, \
            "Beep pattern should be rejected due to low overlap ratio"

    def test_no_false_positives_in_complex_scenario(self):
        """Test that complex cross-matching scenario produces no false positives

        This test verifies the robustness of both algorithms by testing
        all combinations of patterns and audio files where matches shouldn't occur.
        """
        test_cases = [
            # (pattern_file, audio_file, pattern_name)
            ("sample_audios/clips/rthk_beep.wav", "sample_audios/cbs_news_audio_section.wav", "rthk_beep"),
            ("sample_audios/clips/cbs_news.wav", "sample_audios/rthk_section_with_beep.wav", "cbs_news"),
            ("sample_audios/clips/cbs_news_dada.wav", "sample_audios/rthk_section_with_beep.wav", "cbs_news_dada"),
        ]

        for pattern_file, audio_file, pattern_name in test_cases:
            assert Path(pattern_file).exists(), f"Pattern {pattern_file} not found"
            assert Path(audio_file).exists(), f"Audio {audio_file} not found"

            peak_times, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

            assert pattern_name in peak_times, f"{pattern_name} key missing"
            assert len(peak_times[pattern_name]) == 0, \
                f"False positive detected: {pattern_name} in {Path(audio_file).name} " \
                f"produced {len(peak_times[pattern_name])} matches: {peak_times[pattern_name]}"

    def test_correlation_peak_height_threshold(self):
        """Test that peaks below height_min=0.25 are filtered out

        The algorithm only considers correlation peaks with height >= 0.25.
        This test verifies that weak correlations don't produce matches.
        """
        # Mismatched patterns should produce low correlation peaks
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        peak_times, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Should have no matches due to low correlation peaks
        assert len(peak_times['cbs_news']) == 0, \
            "Low correlation peaks should not produce matches"

    def test_verification_stage_filters_false_positives(self):
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
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        peak_times, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Verification stage should reject any potential matches
        assert len(peak_times['rthk_beep']) == 0, \
            "Verification stage should filter out false positives"


class Test16kHzAudioHandling:
    """Tests for handling 16kHz audio files and sample rate conversion"""

    def test_convert_16khz_pattern_to_8khz(self):
        """Test converting 16kHz pattern file to 8kHz format

        The convert function should properly downsample 16kHz audio to 8kHz
        for use as pattern files.
        """
        input_file = "sample_audios/test_16khz/clips/rthk_beep_16k.wav"

        assert Path(input_file).exists(), f"16kHz input file {input_file} not found"

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_file = tmp.name

        try:
            # Convert 16kHz to 8kHz
            convert_audio_to_clip_format(input_file, output_file)

            # Verify output file was created
            assert Path(output_file).exists(), "Converted file was not created"

            # Verify output is 8kHz using ffprobe
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries',
                 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
                 output_file],
                capture_output=True,
                text=True
            )

            sample_rate = int(result.stdout.strip())
            assert sample_rate == 8000, f"Expected 8kHz output, got {sample_rate}Hz"

        finally:
            # Clean up
            if Path(output_file).exists():
                os.unlink(output_file)

    def test_match_16khz_audio_with_8khz_pattern(self):
        """Test matching 16kHz audio file against 8kHz pattern

        The match_pattern function should automatically convert 16kHz audio
        to 8kHz during processing and correctly identify patterns.
        """
        # Use 8kHz pattern (pre-converted)
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        # Use 16kHz audio file
        audio_file = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"

        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"16kHz audio file {audio_file} not found"

        # Run pattern matching - should handle conversion automatically
        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Should find the same matches as with 8kHz audio
        assert 'rthk_beep' in peak_times
        assert len(peak_times['rthk_beep']) == 2, \
            f"Expected 2 matches in 16kHz audio, found {len(peak_times['rthk_beep'])}"

        # Verify timestamps are similar (allowing small variation due to resampling)
        expected_times = [1.4165, 2.419125]
        for i, (actual, expected) in enumerate(zip(sorted(peak_times['rthk_beep']), expected_times)):
            assert abs(actual - expected) < 0.05, \
                f"Match {i}: Expected ~{expected}s, got {actual}s (tolerance increased for resampling)"

    def test_match_16khz_cbs_news(self):
        """Test matching 16kHz CBS news audio against 8kHz pattern

        Tests normal pattern detection with 16kHz audio.
        """
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/test_16khz/cbs_news_audio_section_16k.wav"

        assert Path(pattern_file).exists(), f"Pattern file {pattern_file} not found"
        assert Path(audio_file).exists(), f"16kHz audio file {audio_file} not found"

        peak_times, total_time = match_pattern(audio_file, [pattern_file], debug_mode=False)

        assert 'cbs_news' in peak_times
        assert len(peak_times['cbs_news']) == 1, \
            f"Expected 1 match in 16kHz audio, found {len(peak_times['cbs_news'])}"

        # Verify timestamp (with increased tolerance for resampling)
        expected_time = 25.89875
        actual_time = peak_times['cbs_news'][0]
        assert abs(actual_time - expected_time) < 0.05, \
            f"Expected ~{expected_time}s, got {actual_time}s"

    def test_match_16khz_with_converted_16khz_pattern(self):
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
            # Step 1: Convert pattern from 16kHz to 8kHz
            convert_audio_to_clip_format(input_pattern, converted_pattern)

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

    def test_multiple_16khz_patterns(self):
        """Test matching with multiple 16kHz-sourced patterns

        Tests that multiple patterns can be used simultaneously
        even when source files were originally 16kHz.
        """
        # Convert multiple 16kHz patterns to 8kHz
        input_patterns = [
            "sample_audios/test_16khz/clips/cbs_news_16k.wav",
            "sample_audios/test_16khz/clips/cbs_news_dada_16k.wav"
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

                convert_audio_to_clip_format(input_file, output_file)
                converted_patterns.append(output_file)

            # Match against 16kHz audio
            audio_file = "sample_audios/test_16khz/cbs_news_audio_section_16k.wav"
            assert Path(audio_file).exists()

            peak_times, _ = match_pattern(audio_file, converted_patterns, debug_mode=False)

            # Verify both patterns found their matches
            assert len(peak_times) == 2, "Expected 2 pattern results"

            # Each pattern should have found 1 match
            for pattern_name, matches in peak_times.items():
                assert len(matches) == 1, \
                    f"Pattern {pattern_name} should have 1 match, found {len(matches)}"

        finally:
            # Clean up all temp files
            for temp_file in temp_files:
                if Path(temp_file).exists():
                    os.unlink(temp_file)

    def test_16khz_no_false_positives(self):
        """Test that 16kHz audio doesn't produce false positives

        Verifies that sample rate conversion doesn't introduce
        false positive matches.
        """
        # Use CBS pattern with RTHK audio (should not match)
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"

        assert Path(pattern_file).exists()
        assert Path(audio_file).exists()

        peak_times, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        assert 'cbs_news' in peak_times
        assert len(peak_times['cbs_news']) == 0, \
            f"16kHz conversion should not introduce false positives, found {len(peak_times['cbs_news'])} matches"

    def test_16khz_beep_pattern_rejection(self):
        """Test that beep patterns correctly reject mismatches in 16kHz audio

        Verifies beep detection algorithm works correctly after
        sample rate conversion.
        """
        # Use RTHK beep with CBS audio (should not match)
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/test_16khz/cbs_news_audio_section_16k.wav"

        assert Path(pattern_file).exists()
        assert Path(audio_file).exists()

        peak_times, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        assert 'rthk_beep' in peak_times
        assert len(peak_times['rthk_beep']) == 0, \
            "Beep algorithm should reject mismatches in 16kHz audio"

    def test_sample_rate_preservation_in_results(self):
        """Test that timestamps are correctly adjusted for sample rate conversion

        When 16kHz audio is converted to 8kHz, timestamps should still
        reflect the original audio timeline, not the converted timeline.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"

        # Test with both 8kHz and 16kHz versions of the same audio
        audio_8k = "sample_audios/rthk_section_with_beep.wav"
        audio_16k = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"

        assert Path(pattern_file).exists()
        assert Path(audio_8k).exists()
        assert Path(audio_16k).exists()

        # Match against 8kHz audio
        results_8k, _ = match_pattern(audio_8k, [pattern_file], debug_mode=False)

        # Match against 16kHz audio
        results_16k, _ = match_pattern(audio_16k, [pattern_file], debug_mode=False)

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

    def test_convert_nonexistent_file(self):
        """Test error handling when converting nonexistent file"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_file = tmp.name

        try:
            with pytest.raises(ValueError, match="does not exist"):
                convert_audio_to_clip_format("nonexistent_16k.wav", output_file)
        finally:
            if Path(output_file).exists():
                os.unlink(output_file)


class TestStreamingAudioProcessing:
    """Tests for streaming audio processing using AudioStream and AudioPatternDetector"""

    def test_streaming_rthk_beep_detection(self):
        """Test streaming detection of RTHK beep pattern

        Uses AudioStream and AudioPatternDetector directly to test
        the streaming chunk-based processing.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        assert Path(pattern_file).exists()
        assert Path(audio_file).exists()

        # Load pattern clip
        pattern_clip = AudioClip.from_audio_file(pattern_file)

        # Process audio using streaming
        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
            audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

            # Run detection
            detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
            peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        # Verify results
        assert 'rthk_beep' in peak_times
        assert len(peak_times['rthk_beep']) == 2, \
            f"Expected 2 matches, found {len(peak_times['rthk_beep'])}"

        expected_times = [1.4165, 2.419125]
        for i, (actual, expected) in enumerate(zip(sorted(peak_times['rthk_beep']), expected_times)):
            assert abs(actual - expected) < 0.01, \
                f"Match {i}: Expected ~{expected}s, got {actual}s"

    def test_streaming_cbs_news_detection(self):
        """Test streaming detection of CBS news pattern

        Tests normal pattern detection through the streaming interface.
        """
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        assert Path(pattern_file).exists()
        assert Path(audio_file).exists()

        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
            audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

            detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
            peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'cbs_news' in peak_times
        assert len(peak_times['cbs_news']) == 1

        expected_time = 25.89875
        assert abs(peak_times['cbs_news'][0] - expected_time) < 0.01

    def test_streaming_multiple_patterns(self):
        """Test streaming detection with multiple patterns simultaneously

        Verifies that multiple AudioClips can be processed in a single stream.
        """
        pattern_files = [
            "sample_audios/clips/cbs_news.wav",
            "sample_audios/clips/cbs_news_dada.wav"
        ]
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        assert Path(audio_file).exists()

        # Load multiple pattern clips
        pattern_clips = []
        for pf in pattern_files:
            assert Path(pf).exists()
            pattern_clips.append(AudioClip.from_audio_file(pf))

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
            audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

            detector = AudioPatternDetector(debug_mode=False, audio_clips=pattern_clips)
            peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        # Both patterns should be found
        assert 'cbs_news' in peak_times
        assert 'cbs_news_dada' in peak_times
        assert len(peak_times['cbs_news']) == 1
        assert len(peak_times['cbs_news_dada']) == 1

    def test_streaming_16khz_audio_conversion(self):
        """Test streaming with 16kHz audio auto-conversion

        Verifies that ffmpeg_get_16bit_pcm correctly converts 16kHz to 8kHz
        during streaming and pattern detection works correctly.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav"

        assert Path(pattern_file).exists()
        assert Path(audio_file).exists()

        pattern_clip = AudioClip.from_audio_file(pattern_file)

        # Stream 16kHz audio with conversion to 8kHz
        sr = TARGET_SAMPLE_RATE  # 8000 Hz
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
            audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

            detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
            peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        # Should find same matches as with native 8kHz audio
        assert 'rthk_beep' in peak_times
        assert len(peak_times['rthk_beep']) == 2

        expected_times = [1.4165, 2.419125]
        for i, (actual, expected) in enumerate(zip(sorted(peak_times['rthk_beep']), expected_times)):
            assert abs(actual - expected) < 0.05, \
                f"Match {i}: Expected ~{expected}s, got {actual}s"

    def test_streaming_chunk_processing(self):
        """Test that streaming processes audio in chunks correctly

        Verifies the chunked processing maintains accuracy when
        pattern spans chunk boundaries.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
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

    def test_streaming_small_chunk_size(self):
        """Test streaming with smaller chunk size

        Tests that the detector handles smaller chunks correctly.
        Note: With small chunk sizes and sliding window overlap,
        the same pattern may be detected in multiple chunks,
        resulting in duplicate timestamps. This tests that the
        correct timestamps ARE found (duplicates may exist).
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
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
        expected_times = [1.4165, 2.419125]
        found_times = set()
        for actual in peak_times['rthk_beep']:
            for expected in expected_times:
                if abs(actual - expected) < 0.01:
                    found_times.add(expected)
                    break

        assert len(found_times) == len(expected_times), \
            f"Expected to find timestamps near {expected_times}, found {peak_times['rthk_beep']}"

    def test_streaming_no_match_scenario(self):
        """Test streaming when pattern is not present in audio

        Verifies that streaming processing correctly returns empty results.
        """
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
            audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

            detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
            peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'cbs_news' in peak_times
        assert len(peak_times['cbs_news']) == 0

    def test_streaming_total_time_accuracy(self):
        """Test that total_time accurately reflects processed audio duration

        Verifies the streaming processor correctly tracks total audio processed.
        """
        audio_file = "sample_audios/rthk_section_with_beep.wav"
        pattern_file = "sample_audios/clips/rthk_beep.wav"

        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
            audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

            detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
            peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        # rthk_section_with_beep.wav is ~4.08 seconds
        assert 4.0 < total_time < 4.2, f"Expected ~4.08s, got {total_time}s"

    def test_audio_clip_from_file(self):
        """Test AudioClip.from_audio_file correctly loads pattern files

        Verifies AudioClip dataclass is properly initialized.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"

        clip = AudioClip.from_audio_file(pattern_file)

        assert clip.name == "rthk_beep"
        assert clip.sample_rate == TARGET_SAMPLE_RATE
        assert len(clip.audio) > 0
        assert clip.clip_length_seconds() > 0

    def test_audio_clip_sample_rate_validation(self):
        """Test that AudioPatternDetector validates pattern sample rates

        Patterns must match TARGET_SAMPLE_RATE (8000 Hz).
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"

        # Create clip with correct sample rate
        pattern_clip = AudioClip.from_audio_file(pattern_file)

        # Verify clip has correct sample rate
        assert pattern_clip.sample_rate == TARGET_SAMPLE_RATE

        # Detector should accept valid clips
        detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
        assert len(detector.audio_clips) == 1

    def test_streaming_maintains_pattern_order(self):
        """Test that multiple patterns maintain their result order

        Verifies each pattern's results are stored under the correct key.
        """
        pattern_files = [
            "sample_audios/clips/rthk_beep.wav",
            "sample_audios/clips/cbs_news.wav",
            "sample_audios/clips/cbs_news_dada.wav"
        ]
        audio_file = "sample_audios/cbs_news_audio_section.wav"

        pattern_clips = [AudioClip.from_audio_file(pf) for pf in pattern_files]

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
            audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

            detector = AudioPatternDetector(debug_mode=False, audio_clips=pattern_clips)
            peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        # All patterns should have result entries
        assert 'rthk_beep' in peak_times
        assert 'cbs_news' in peak_times
        assert 'cbs_news_dada' in peak_times

        # RTHK beep shouldn't match CBS audio
        assert len(peak_times['rthk_beep']) == 0
        # CBS patterns should match
        assert len(peak_times['cbs_news']) == 1
        assert len(peak_times['cbs_news_dada']) == 1

    def test_streaming_duplicate_pattern_names_rejected(self):
        """Test that duplicate pattern names are rejected

        AudioPatternDetector should raise ValueError for duplicate clip names.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"

        clip1 = AudioClip.from_audio_file(pattern_file)
        clip2 = AudioClip.from_audio_file(pattern_file)  # Same name

        with pytest.raises(ValueError, match="needs to be unique"):
            AudioPatternDetector(debug_mode=False, audio_clips=[clip1, clip2])

    def test_streaming_16khz_cbs_news(self):
        """Test streaming 16kHz CBS news audio with conversion

        Tests normal pattern detection through streaming with sample rate conversion.
        """
        pattern_file = "sample_audios/clips/cbs_news.wav"
        audio_file = "sample_audios/test_16khz/cbs_news_audio_section_16k.wav"

        assert Path(pattern_file).exists()
        assert Path(audio_file).exists()

        pattern_clip = AudioClip.from_audio_file(pattern_file)

        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
            audio_stream = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)

            detector = AudioPatternDetector(debug_mode=False, audio_clips=[pattern_clip])
            peak_times, total_time = detector.find_clip_in_audio(audio_stream)

        assert 'cbs_news' in peak_times
        assert len(peak_times['cbs_news']) == 1

        expected_time = 25.89875
        assert abs(peak_times['cbs_news'][0] - expected_time) < 0.05

    def test_streaming_results_match_high_level_api(self):
        """Test that streaming results match the high-level match_pattern API

        Ensures consistency between low-level streaming and high-level APIs.
        """
        pattern_file = "sample_audios/clips/rthk_beep.wav"
        audio_file = "sample_audios/rthk_section_with_beep.wav"

        # High-level API result
        high_level_results, _ = match_pattern(audio_file, [pattern_file], debug_mode=False)

        # Low-level streaming result
        pattern_clip = AudioClip.from_audio_file(pattern_file)
        sr = TARGET_SAMPLE_RATE
        with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
            audio_name = Path(audio_file).stem
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