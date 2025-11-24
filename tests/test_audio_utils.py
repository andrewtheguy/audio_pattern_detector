import os
import tempfile

import numpy as np
import pytest

from audio_pattern_detector.audio_utils import load_wave_file, write_wav_file
from audio_pattern_detector.convert import convert_audio_to_clip_format


class TestWriteWavFile:
    def test_write_and_read_roundtrip(self):
        """Test that write_wav_file creates a valid wav file that can be read back."""
        sample_rate = 8000
        duration = 1.0  # 1 second
        # Create a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Write the file
            write_wav_file(temp_path, audio_data, sample_rate)

            # Verify file exists and has content
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0

            # Read it back
            loaded_audio = load_wave_file(temp_path, sample_rate)

            # Check that the data matches (allowing for small float precision differences)
            np.testing.assert_array_almost_equal(audio_data, loaded_audio, decimal=4)
        finally:
            os.unlink(temp_path)

    def test_write_different_sample_rates(self):
        """Test writing files with different sample rates."""
        for sample_rate in [8000, 16000, 44100]:
            audio_data = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            try:
                write_wav_file(temp_path, audio_data, sample_rate)
                assert os.path.exists(temp_path)

                # Read back and verify
                loaded = load_wave_file(temp_path, sample_rate)
                assert len(loaded) == sample_rate
            finally:
                os.unlink(temp_path)


class TestLoadWaveFile:
    def test_load_existing_wav_file(self):
        """Test loading an existing wav file."""
        # Use one of the sample files
        sample_file = "sample_audios/clips/rthk_beep.wav"
        if os.path.exists(sample_file):
            audio = load_wave_file(sample_file, 8000)
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert len(audio) > 0
            # Check normalized range
            assert np.max(np.abs(audio)) <= 1.0

    def test_load_wrong_sample_rate_raises(self):
        """Test that loading with wrong sample rate raises ValueError."""
        sample_file = "sample_audios/clips/rthk_beep.wav"
        if os.path.exists(sample_file):
            with pytest.raises(ValueError, match="sample rate"):
                load_wave_file(sample_file, 44100)  # Wrong sample rate

    def test_load_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises an error."""
        with pytest.raises(ValueError):
            load_wave_file("nonexistent_file.wav", 8000)

    def test_load_stereo_file_raises(self):
        """Test that loading a stereo file raises ValueError."""
        # Create a temporary stereo file
        sample_rate = 8000
        audio_left = np.zeros(sample_rate, dtype=np.int16)
        audio_right = np.zeros(sample_rate, dtype=np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Use ffmpeg to create a stereo file
            import subprocess

            # Create raw stereo PCM data (interleaved)
            stereo_data = np.column_stack((audio_left, audio_right)).flatten()

            cmd = [
                "ffmpeg", "-y",
                "-f", "s16le",
                "-ar", str(sample_rate),
                "-ac", "2",  # Stereo
                "-i", "pipe:",
                "-loglevel", "error",
                temp_path
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            proc.communicate(stereo_data.tobytes())

            with pytest.raises(ValueError, match="not mono"):
                load_wave_file(temp_path, sample_rate)
        finally:
            os.unlink(temp_path)


class TestRoundTrip:
    def test_preserves_audio_content(self):
        """Test that writing and reading preserves audio content."""
        sample_rate = 8000
        # Create audio with various values
        audio_data = np.array([0.0, 0.5, -0.5, 0.99, -0.99, 0.25, -0.25], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            write_wav_file(temp_path, audio_data, sample_rate)
            loaded = load_wave_file(temp_path, sample_rate)

            # Values should be close (16-bit quantization introduces small errors)
            np.testing.assert_array_almost_equal(audio_data, loaded, decimal=4)
        finally:
            os.unlink(temp_path)

    def test_load_sample_file_and_rewrite(self):
        """Test loading a sample file and writing it back."""
        sample_file = "sample_audios/clips/rthk_beep.wav"
        if not os.path.exists(sample_file):
            pytest.skip("Sample file not found")

        sample_rate = 8000
        original = load_wave_file(sample_file, sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            write_wav_file(temp_path, original, sample_rate)
            reloaded = load_wave_file(temp_path, sample_rate)

            # Should be identical
            np.testing.assert_array_almost_equal(original, reloaded, decimal=5)
        finally:
            os.unlink(temp_path)


class TestConvertAudioToClipFormat:
    def _get_audio_info(self, filepath):
        """Helper to get audio file metadata using ffprobe."""
        import json
        import subprocess

        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels,sample_rate,bits_per_sample,duration",
            "-of", "json",
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return {
            "channels": stream.get("channels"),
            "sample_rate": int(stream.get("sample_rate")),
            "bits_per_sample": stream.get("bits_per_sample"),
            "duration": float(stream.get("duration", 0)),
        }

    def test_convert_8khz_file(self):
        """Test converting an 8kHz file produces valid output."""
        input_file = "sample_audios/clips/rthk_beep.wav"
        if not os.path.exists(input_file):
            pytest.skip("Sample file not found")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Get input duration
            input_info = self._get_audio_info(input_file)

            convert_audio_to_clip_format(input_file, output_path)

            # Verify output file exists
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Verify output format
            output_info = self._get_audio_info(output_path)
            assert output_info["channels"] == 1, "Output should be mono"
            assert output_info["sample_rate"] == 8000, "Output should be 8kHz"
            assert output_info["bits_per_sample"] == 16, "Output should be 16-bit"

            # Verify duration is preserved
            assert abs(output_info["duration"] - input_info["duration"]) < 0.01, \
                f"Duration mismatch: input={input_info['duration']}, output={output_info['duration']}"

            # Verify we can load the output with load_wave_file
            audio = load_wave_file(output_path, 8000)
            assert len(audio) > 0
            assert audio.dtype == np.float32
        finally:
            os.unlink(output_path)

    def test_convert_16khz_file_to_8khz(self):
        """Test converting a 16kHz file downsamples to 8kHz."""
        input_file = "sample_audios/test_16khz/clips/rthk_beep_16k.wav"
        if not os.path.exists(input_file):
            pytest.skip("16kHz sample file not found")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Get input duration
            input_info = self._get_audio_info(input_file)

            convert_audio_to_clip_format(input_file, output_path)

            # Verify output format is 8kHz
            output_info = self._get_audio_info(output_path)
            assert output_info["channels"] == 1, "Output should be mono"
            assert output_info["sample_rate"] == 8000, "Output should be 8kHz"
            assert output_info["bits_per_sample"] == 16, "Output should be 16-bit"

            # Verify duration is preserved after resampling
            assert abs(output_info["duration"] - input_info["duration"]) < 0.01, \
                f"Duration mismatch: input={input_info['duration']}, output={output_info['duration']}"

            # Verify we can load the output
            audio = load_wave_file(output_path, 8000)
            assert len(audio) > 0
        finally:
            os.unlink(output_path)

    def test_convert_nonexistent_file_raises(self):
        """Test that converting nonexistent file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            with pytest.raises(ValueError, match="does not exist"):
                convert_audio_to_clip_format("nonexistent_file.wav", output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_convert_preserves_audio_content(self):
        """Test that conversion preserves audio content for same sample rate."""
        input_file = "sample_audios/clips/cbs_news.wav"
        if not os.path.exists(input_file):
            pytest.skip("Sample file not found")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Load original
            original = load_wave_file(input_file, 8000)

            # Convert
            convert_audio_to_clip_format(input_file, output_path)

            # Load converted
            converted = load_wave_file(output_path, 8000)

            # Should have same length and similar content
            assert len(converted) == len(original)
            np.testing.assert_array_almost_equal(original, converted, decimal=4)
        finally:
            os.unlink(output_path)

    def test_output_is_valid_wav(self):
        """Test that output is a valid wav file that can be used as a clip."""
        import json
        import subprocess

        input_file = "sample_audios/rthk_section_with_beep.wav"
        if not os.path.exists(input_file):
            pytest.skip("Sample file not found")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Get input duration
            input_info = self._get_audio_info(input_file)

            convert_audio_to_clip_format(input_file, output_path)

            # Verify the output is actually a wav file using ffprobe
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=format_name",
                "-show_entries", "stream=codec_name",
                "-of", "json",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"ffprobe failed: {result.stderr}"

            probe_data = json.loads(result.stdout)

            # Verify format is wav
            format_name = probe_data.get("format", {}).get("format_name", "")
            assert "wav" in format_name, f"Output is not a wav file, got format: {format_name}"

            # Verify codec is PCM
            codec_name = probe_data.get("streams", [{}])[0].get("codec_name", "")
            assert "pcm_s16le" in codec_name, f"Output is not PCM 16-bit, got codec: {codec_name}"

            # Also verify audio properties
            output_info = self._get_audio_info(output_path)
            assert output_info["channels"] == 1, "Output should be mono"
            assert output_info["sample_rate"] == 8000, "Output should be 8kHz"
            assert output_info["bits_per_sample"] == 16, "Output should be 16-bit"

            # Verify duration is preserved
            assert abs(output_info["duration"] - input_info["duration"]) < 0.01, \
                f"Duration mismatch: input={input_info['duration']}, output={output_info['duration']}"

            # Verify the output can be loaded and used
            audio = load_wave_file(output_path, 8000)

            # Basic sanity checks
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert len(audio) > 0
            # Audio should be in normalized range
            assert np.max(audio) <= 1.0
            assert np.min(audio) >= -1.0
        finally:
            os.unlink(output_path)
