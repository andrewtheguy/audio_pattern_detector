"""CLI integration tests for audio-pattern-detector.

Tests the command-line interface to verify:
1. Arguments are parsed and passed correctly to internal functions
2. Output format is correct (JSON vs JSONL)
3. Error handling for CLI-specific errors

Internal detection logic is tested in test_detector_api.py and other test files.
These tests focus on CLI argument passing, not duplicating detection logic tests.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path



def run_cli(*args, stdin_data=None, check=True):
    """Run the CLI with given arguments and return result."""
    cmd = ["uv", "run", "audio-pattern-detector", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=stdin_data,
        check=False,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


def run_cli_binary(*args, stdin_data=None, check=True):
    """Run the CLI with binary stdin data."""
    cmd = ["uv", "run", "audio-pattern-detector", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        input=stdin_data,
        check=False,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


# --- Help and Basic CLI Tests ---


def test_cli_help():
    """Test that --help works."""
    result = run_cli("--help")
    assert result.returncode == 0
    assert "audio-pattern-detector" in result.stdout
    assert "convert" in result.stdout
    assert "match" in result.stdout
    assert "show-config" in result.stdout


def test_cli_match_help():
    """Test that match --help shows all options."""
    result = run_cli("match", "--help")
    assert result.returncode == 0
    assert "--pattern-file" in result.stdout
    assert "--pattern-folder" in result.stdout
    assert "--audio-file" in result.stdout
    assert "--audio-folder" in result.stdout
    assert "--stdin" in result.stdout
    assert "--raw-pcm" in result.stdout
    assert "--source-sample-rate" in result.stdout
    assert "--target-sample-rate" in result.stdout
    assert "--jsonl" in result.stdout
    assert "--chunk-seconds" in result.stdout


def test_cli_convert_help():
    """Test that convert --help shows options."""
    result = run_cli("convert", "--help")
    assert result.returncode == 0
    assert "--audio-file" in result.stdout
    assert "--dest-file" in result.stdout


def test_cli_show_config_help():
    """Test that show-config --help shows options."""
    result = run_cli("show-config", "--help")
    assert result.returncode == 0
    assert "--pattern-file" in result.stdout
    assert "--pattern-folder" in result.stdout


def test_cli_no_command():
    """Test CLI with no command shows help."""
    result = run_cli(check=False)
    assert result.returncode == 1


# --- Match Command: Argument Passing Tests ---


def test_match_audio_file_returns_json():
    """Test match with --audio-file returns valid JSON output."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
    )
    assert result.returncode == 0

    # Verify output is valid JSON with expected structure
    output = json.loads(result.stdout)
    assert isinstance(output, dict)
    assert "rthk_beep" in output
    assert isinstance(output["rthk_beep"], list)


def test_match_pattern_folder_passes_multiple_patterns():
    """Test --pattern-folder passes all patterns to detector."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/cbs_news_audio_section.wav",
        "--pattern-folder", "sample_audios/clips",
    )
    assert result.returncode == 0

    output = json.loads(result.stdout)

    # All patterns from folder should be in output
    assert "cbs_news" in output
    assert "cbs_news_dada" in output
    assert "rthk_beep" in output


def test_match_chunk_seconds_argument_passed():
    """Test --chunk-seconds argument is passed to detector."""
    # Using auto mode - should work without error
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--chunk-seconds", "auto",
    )
    assert result.returncode == 0

    # Using explicit value
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--chunk-seconds", "10",
    )
    assert result.returncode == 0


def test_match_chunk_seconds_invalid_value():
    """Test --chunk-seconds rejects invalid values."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--chunk-seconds", "invalid",
        check=False,
    )
    assert result.returncode != 0
    assert "auto" in result.stderr or "integer" in result.stderr


# --- Match Command: --stdin Tests (WAV and Raw PCM, Always JSONL) ---


def _convert_wav_to_raw_pcm(wav_file: str, sample_rate: int = 8000) -> bytes:
    """Helper to convert WAV file to raw float32 PCM."""
    cmd = [
        "ffmpeg", "-i", wav_file,
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate),
        "-loglevel", "error", "pipe:"
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


def _convert_file_to_wav_stdout(audio_file: str, sample_rate: int = 8000) -> bytes:
    """Helper to convert audio file to WAV and pipe to stdout."""
    cmd = [
        "ffmpeg", "-i", audio_file,
        "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate),
        "-loglevel", "error", "pipe:"
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


def test_match_stdin_reads_wav():
    """Test match reads WAV from stdin and outputs JSONL."""
    wav_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav")

    result = run_cli_binary(
        "match",
        "--stdin",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        stdin_data=wav_data,
    )
    assert result.returncode == 0

    # stdin mode always outputs JSONL
    lines = result.stdout.decode().strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should have start and end events
    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "end"

    # Should have pattern_detected events
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) > 0
    assert pattern_events[0]["clip_name"] == "rthk_beep"


def test_match_stdin_reads_raw_pcm():
    """Test match --raw-pcm reads raw PCM from stdin and outputs JSONL."""
    raw_pcm_data = _convert_wav_to_raw_pcm("sample_audios/rthk_section_with_beep.wav")

    result = run_cli_binary(
        "match",
        "--stdin",
        "--raw-pcm",
        "--source-sample-rate", "8000",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        stdin_data=raw_pcm_data,
    )
    assert result.returncode == 0

    # stdin mode always outputs JSONL
    lines = result.stdout.decode().strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should have start and end events
    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "end"

    # Should have pattern_detected events
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) > 0
    assert pattern_events[0]["clip_name"] == "rthk_beep"


def test_match_stdin_with_pattern_folder():
    """Test --stdin with WAV works with --pattern-folder."""
    wav_data = _convert_file_to_wav_stdout("sample_audios/cbs_news_audio_section.wav")

    result = run_cli_binary(
        "match",
        "--stdin",
        "--pattern-folder", "sample_audios/clips",
        stdin_data=wav_data,
    )
    assert result.returncode == 0

    # stdin mode always outputs JSONL
    lines = result.stdout.decode().strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Collect all detected pattern names
    pattern_names = {e["clip_name"] for e in events if e["type"] == "pattern_detected"}

    # CBS news patterns should be detected in CBS audio
    assert "cbs_news" in pattern_names or "cbs_news_dada" in pattern_names


# --- Match Command: --jsonl Output Format Tests ---


def test_match_jsonl_output_format():
    """Test --jsonl produces correct JSONL output format."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--jsonl",
    )
    assert result.returncode == 0

    # Each line should be valid JSON
    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    # First event should be "start"
    assert events[0]["type"] == "start"
    assert "source" in events[0]

    # Last event should be "end"
    assert events[-1]["type"] == "end"
    assert "total_time" in events[-1]
    assert "total_time_formatted" in events[-1]

    # Middle events should be "pattern_detected"
    for event in events[1:-1]:
        assert event["type"] == "pattern_detected"
        assert "clip_name" in event
        assert "timestamp" in event
        assert "timestamp_formatted" in event


def test_match_jsonl_start_event_source():
    """Test --jsonl start event contains correct source."""
    # Test with audio file
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--jsonl",
    )
    start_event = json.loads(result.stdout.strip().split("\n")[0])
    assert "rthk_section_with_beep.wav" in start_event["source"]

    # Test with stdin (WAV mode, always outputs JSONL)
    wav_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav")

    result = run_cli_binary(
        "match",
        "--stdin",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        stdin_data=wav_data,
    )
    start_event = json.loads(result.stdout.decode().strip().split("\n")[0])
    assert start_event["source"] == "stdin"


def test_match_jsonl_no_match_only_start_end():
    """Test --jsonl output when no patterns match."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/cbs_news.wav",
        "--jsonl",
    )
    assert result.returncode == 0

    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should only have start and end events (no pattern_detected)
    assert len(events) == 2
    assert events[0]["type"] == "start"
    assert events[1]["type"] == "end"


# --- Show-config Command Tests ---


def test_show_config_returns_json():
    """Test show-config returns valid JSON with expected structure."""
    result = run_cli(
        "show-config",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
    )
    assert result.returncode == 0

    config = json.loads(result.stdout)
    assert "default_seconds_per_chunk" in config
    assert "min_chunk_size_seconds" in config
    assert "sample_rate" in config
    assert "clips" in config
    assert "rthk_beep" in config["clips"]


def test_show_config_with_pattern_folder():
    """Test show-config with --pattern-folder includes all patterns."""
    result = run_cli(
        "show-config",
        "--pattern-folder", "sample_audios/clips",
    )
    assert result.returncode == 0

    config = json.loads(result.stdout)
    assert "cbs_news" in config["clips"]
    assert "cbs_news_dada" in config["clips"]
    assert "rthk_beep" in config["clips"]


def test_show_config_clip_info():
    """Test show-config returns per-clip configuration."""
    result = run_cli(
        "show-config",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
    )
    config = json.loads(result.stdout)

    clip_config = config["clips"]["rthk_beep"]
    assert "duration_seconds" in clip_config
    assert "sliding_window_seconds" in clip_config
    assert "is_pure_tone" in clip_config


# --- Convert Command Tests ---


def test_convert_creates_output_file():
    """Test convert command creates output file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_file = tmp.name

    try:
        result = run_cli(
            "convert",
            "--audio-file", "sample_audios/test_16khz/rthk_section_with_beep_16k.wav",
            "--dest-file", output_file,
        )
        assert result.returncode == 0
        assert Path(output_file).exists()

        # Verify output is 8kHz
        probe_result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", output_file],
            capture_output=True,
            text=True,
        )
        sample_rate = int(probe_result.stdout.strip())
        assert sample_rate == 8000
    finally:
        if Path(output_file).exists():
            os.unlink(output_file)


def test_convert_output_usable_as_pattern():
    """Test converted file can be used as pattern for matching."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        converted_pattern = tmp.name

    try:
        # Convert 16kHz pattern to 8kHz
        run_cli(
            "convert",
            "--audio-file", "sample_audios/test_16khz/clips/rthk_beep_16k.wav",
            "--dest-file", converted_pattern,
        )

        # Use converted pattern for matching
        result = run_cli(
            "match",
            "--audio-file", "sample_audios/rthk_section_with_beep.wav",
            "--pattern-file", converted_pattern,
        )
        assert result.returncode == 0

        output = json.loads(result.stdout)
        pattern_name = Path(converted_pattern).stem
        assert pattern_name in output
        # Should find matches
        assert len(output[pattern_name]) > 0
    finally:
        if Path(converted_pattern).exists():
            os.unlink(converted_pattern)


# --- Error Handling Tests ---


def test_match_nonexistent_audio_file():
    """Test match with nonexistent audio file."""
    result = run_cli(
        "match",
        "--audio-file", "nonexistent.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        check=False,
    )
    assert result.returncode != 0


def test_match_nonexistent_pattern_file():
    """Test match with nonexistent pattern file."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "nonexistent.wav",
        check=False,
    )
    assert result.returncode != 0


def test_match_no_audio_source():
    """Test match without any audio source."""
    result = run_cli(
        "match",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        check=False,
    )
    assert result.returncode != 0
    assert "Please provide" in result.stderr


def test_match_no_pattern():
    """Test match without any pattern."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        check=False,
    )
    assert result.returncode != 0
    assert "Please provide" in result.stderr


def test_convert_nonexistent_file():
    """Test convert with nonexistent input file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_file = tmp.name

    try:
        result = run_cli(
            "convert",
            "--audio-file", "nonexistent.wav",
            "--dest-file", output_file,
            check=False,
        )
        assert result.returncode != 0
    finally:
        if Path(output_file).exists():
            os.unlink(output_file)


def test_show_config_no_pattern():
    """Test show-config without any pattern."""
    result = run_cli(
        "show-config",
        check=False,
    )
    assert result.returncode != 0
    assert "Please provide" in result.stderr


def test_show_config_nonexistent_pattern():
    """Test show-config with nonexistent pattern file."""
    result = run_cli(
        "show-config",
        "--pattern-file", "nonexistent.wav",
        check=False,
    )
    assert result.returncode != 0


# --- 16kHz Audio Auto-Conversion Tests ---


def test_match_16khz_audio_auto_converts():
    """Test match with 16kHz audio file (auto-converted to 8kHz)."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/test_16khz/rthk_section_with_beep_16k.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
    )
    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert "rthk_beep" in output
    # Should find matches despite sample rate difference
    assert len(output["rthk_beep"]) > 0


# --- Stdin with Sample Rate Tests ---


def test_stdin_raw_pcm_with_source_sample_rate_resamples():
    """Test --stdin --raw-pcm with --source-sample-rate resamples audio correctly."""
    import numpy as np

    # Generate raw float32 PCM at 16kHz using ffmpeg
    raw_pcm_data = _convert_wav_to_raw_pcm("sample_audios/rthk_section_with_beep.wav", sample_rate=16000)

    # Verify we have valid float32 data at 16kHz
    audio = np.frombuffer(raw_pcm_data, dtype=np.float32)
    assert len(audio) > 0

    # Use stdin mode with 16kHz raw PCM (will be resampled to 8kHz)
    result = run_cli_binary(
        "match",
        "--stdin",
        "--raw-pcm",
        "--source-sample-rate", "16000",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        stdin_data=raw_pcm_data,
    )
    assert result.returncode == 0

    # stdin mode always outputs JSONL
    lines = result.stdout.decode().strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should have pattern_detected events despite sample rate conversion
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) > 0


def test_stdin_wav_with_different_sample_rate_resamples():
    """Test --stdin WAV mode automatically resamples from header sample rate."""
    # Generate WAV at 16kHz
    wav_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav", sample_rate=16000)

    # Use stdin mode with WAV (sample rate read from header, resampled to 8kHz)
    result = run_cli_binary(
        "match",
        "--stdin",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        stdin_data=wav_data,
    )
    assert result.returncode == 0

    # stdin mode always outputs JSONL
    lines = result.stdout.decode().strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should have pattern_detected events despite sample rate conversion
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) > 0


def test_stdin_raw_pcm_requires_source_sample_rate():
    """Test --stdin --raw-pcm requires --source-sample-rate."""
    raw_pcm_data = _convert_wav_to_raw_pcm("sample_audios/rthk_section_with_beep.wav")

    result = run_cli_binary(
        "match",
        "--stdin",
        "--raw-pcm",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        stdin_data=raw_pcm_data,
        check=False,
    )
    assert result.returncode != 0
    assert b"--source-sample-rate is required" in result.stderr


def test_stdin_source_sample_rate_only_with_raw_pcm():
    """Test --source-sample-rate can only be used with --raw-pcm."""
    wav_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav")

    result = run_cli_binary(
        "match",
        "--stdin",
        "--source-sample-rate", "8000",  # Not allowed without --raw-pcm
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        stdin_data=wav_data,
        check=False,
    )
    assert result.returncode != 0
    assert b"--source-sample-rate can only be used with --raw-pcm" in result.stderr


def test_stdin_raw_pcm_rejects_wav_input():
    """Test --raw-pcm mode detects and rejects WAV input with helpful error."""
    # Send WAV data to raw PCM mode - should be rejected
    wav_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav")

    result = run_cli_binary(
        "match",
        "--stdin",
        "--raw-pcm",
        "--source-sample-rate", "8000",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        stdin_data=wav_data,
        check=False,
    )
    assert result.returncode != 0
    assert b"WAV format" in result.stderr or b"RIFF/WAVE" in result.stderr


# --- Match Command: --multiplexed-stdin Tests ---


def _build_multiplexed_payload(patterns: list[tuple[str, bytes]], audio_data: bytes) -> bytes:
    """Build a multiplexed stdin payload.

    Protocol:
        [4 bytes] number_of_patterns (uint32 little-endian)
        For each pattern:
            [4 bytes] name_length (uint32 little-endian)
            [name_length bytes] name (UTF-8)
            [4 bytes] data_length (uint32 little-endian)
            [data_length bytes] WAV data
        [remaining bytes] audio stream
    """
    payload = bytearray()

    # Number of patterns
    payload.extend(len(patterns).to_bytes(4, byteorder='little', signed=False))

    # Each pattern
    for name, wav_data in patterns:
        name_bytes = name.encode('utf-8')
        payload.extend(len(name_bytes).to_bytes(4, byteorder='little', signed=False))
        payload.extend(name_bytes)
        payload.extend(len(wav_data).to_bytes(4, byteorder='little', signed=False))
        payload.extend(wav_data)

    # Audio stream
    payload.extend(audio_data)

    return bytes(payload)


def test_multiplexed_stdin_help():
    """Test that --multiplexed-stdin is shown in help."""
    result = run_cli("match", "--help")
    assert "--multiplexed-stdin" in result.stdout


def test_multiplexed_stdin_single_pattern_wav_audio():
    """Test --multiplexed-stdin with single pattern and WAV audio."""
    # Load pattern WAV
    with open("sample_audios/clips/rthk_beep.wav", "rb") as f:
        pattern_data = f.read()

    # Convert audio to WAV
    audio_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav")

    # Build multiplexed payload
    payload = _build_multiplexed_payload(
        patterns=[("rthk_beep", pattern_data)],
        audio_data=audio_data,
    )

    result = run_cli_binary(
        "match",
        "--multiplexed-stdin",
        stdin_data=payload,
    )
    assert result.returncode == 0

    # multiplexed-stdin mode always outputs JSONL
    lines = result.stdout.decode().strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should have start and end events
    assert events[0]["type"] == "start"
    assert events[0]["source"] == "multiplexed-stdin"
    assert events[-1]["type"] == "end"

    # Should have pattern_detected events
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) > 0
    assert pattern_events[0]["clip_name"] == "rthk_beep"


def test_multiplexed_stdin_multiple_patterns():
    """Test --multiplexed-stdin with multiple patterns."""
    # Load pattern WAVs
    with open("sample_audios/clips/rthk_beep.wav", "rb") as f:
        pattern1_data = f.read()
    with open("sample_audios/clips/cbs_news.wav", "rb") as f:
        pattern2_data = f.read()

    # Convert audio to WAV (this audio has RTHK beep but not CBS)
    audio_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav")

    # Build multiplexed payload with two patterns
    payload = _build_multiplexed_payload(
        patterns=[
            ("rthk_beep", pattern1_data),
            ("cbs_news", pattern2_data),
        ],
        audio_data=audio_data,
    )

    result = run_cli_binary(
        "match",
        "--multiplexed-stdin",
        stdin_data=payload,
    )
    assert result.returncode == 0

    lines = result.stdout.decode().strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should detect rthk_beep but not cbs_news
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    clip_names = {e["clip_name"] for e in pattern_events}
    assert "rthk_beep" in clip_names


def test_multiplexed_stdin_with_raw_pcm_audio():
    """Test --multiplexed-stdin with --raw-pcm for audio stream."""
    # Load pattern WAV
    with open("sample_audios/clips/rthk_beep.wav", "rb") as f:
        pattern_data = f.read()

    # Convert audio to raw PCM
    audio_data = _convert_wav_to_raw_pcm("sample_audios/rthk_section_with_beep.wav")

    # Build multiplexed payload
    payload = _build_multiplexed_payload(
        patterns=[("rthk_beep", pattern_data)],
        audio_data=audio_data,
    )

    result = run_cli_binary(
        "match",
        "--multiplexed-stdin",
        "--raw-pcm",
        "--source-sample-rate", "8000",
        stdin_data=payload,
    )
    assert result.returncode == 0

    lines = result.stdout.decode().strip().split("\n")
    events = [json.loads(line) for line in lines]

    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "end"

    # Should have pattern_detected events
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) > 0
    assert pattern_events[0]["clip_name"] == "rthk_beep"


def test_multiplexed_stdin_requires_no_pattern_file():
    """Test --multiplexed-stdin does not require --pattern-file."""
    # Load pattern WAV
    with open("sample_audios/clips/rthk_beep.wav", "rb") as f:
        pattern_data = f.read()

    # Convert audio to WAV
    audio_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav")

    # Build multiplexed payload
    payload = _build_multiplexed_payload(
        patterns=[("test_pattern", pattern_data)],
        audio_data=audio_data,
    )

    # Should work without --pattern-file or --pattern-folder
    result = run_cli_binary(
        "match",
        "--multiplexed-stdin",
        stdin_data=payload,
    )
    assert result.returncode == 0


def test_multiplexed_stdin_raw_pcm_requires_source_sample_rate():
    """Test --multiplexed-stdin with --raw-pcm requires --source-sample-rate."""
    with open("sample_audios/clips/rthk_beep.wav", "rb") as f:
        pattern_data = f.read()

    audio_data = _convert_wav_to_raw_pcm("sample_audios/rthk_section_with_beep.wav")
    payload = _build_multiplexed_payload(
        patterns=[("test", pattern_data)],
        audio_data=audio_data,
    )

    result = run_cli_binary(
        "match",
        "--multiplexed-stdin",
        "--raw-pcm",
        # Missing --source-sample-rate
        stdin_data=payload,
        check=False,
    )
    assert result.returncode != 0
    assert b"--source-sample-rate is required" in result.stderr
