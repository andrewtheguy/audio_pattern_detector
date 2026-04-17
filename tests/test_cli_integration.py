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
    assert "match" in result.stdout
    assert "show-config" in result.stdout


def test_cli_match_help():
    """Test that match --help shows all options."""
    result = run_cli("match", "--help")
    assert result.returncode == 0
    assert "--pattern-file" in result.stdout
    assert "--pattern-folder" in result.stdout
    assert "--stdin" in result.stdout
    assert "--target-sample-rate" in result.stdout
    assert "--chunk-seconds" in result.stdout


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


def test_match_audio_file_returns_jsonl():
    """Test match with positional audio file returns JSONL output."""
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
    )
    assert result.returncode == 0

    # Verify output is valid JSONL with expected events
    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "end"

    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) > 0
    assert pattern_events[0]["clip_name"] == "rthk_beep"


def test_match_pattern_folder_passes_multiple_patterns():
    """Test --pattern-folder passes all patterns to detector."""
    result = run_cli(
        "match",
        "sample_audios/cbs_news_audio_section.wav",
        "--pattern-folder", "sample_audios/clips",
    )
    assert result.returncode == 0

    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    # All patterns from folder should appear in pattern_detected events
    clip_names = {e["clip_name"] for e in events if e["type"] == "pattern_detected"}
    assert "cbs_news" in clip_names


def test_match_chunk_seconds_argument_passed():
    """Test --chunk-seconds argument is passed to detector."""
    # Using auto mode - should work without error
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
        "--chunk-seconds", "auto",
    )
    assert result.returncode == 0

    # Using explicit value
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
        "--chunk-seconds", "10",
    )
    assert result.returncode == 0


def test_match_chunk_seconds_invalid_value():
    """Test --chunk-seconds rejects invalid values."""
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
        "--chunk-seconds", "invalid",
        check=False,
    )
    assert result.returncode != 0
    assert "auto" in result.stderr or "integer" in result.stderr


# --- Match Command: --stdin Tests (WAV, Always JSONL) ---


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
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
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
    assert "timestamp_ms" in pattern_events[0]
    assert isinstance(pattern_events[0]["timestamp_ms"], int)
    assert "timestamp_formatted" in pattern_events[0]
    assert isinstance(pattern_events[0]["timestamp_formatted"], str)
    assert "total_time_ms" in events[-1]
    assert isinstance(events[-1]["total_time_ms"], int)
    assert "total_time_formatted" in events[-1]
    assert isinstance(events[-1]["total_time_formatted"], str)


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
    assert "cbs_news" in pattern_names


# --- Match Command: JSONL Output Format Tests ---


def test_match_jsonl_output_format():
    """Test JSONL output includes both timestamp formats by default."""
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
    )
    assert result.returncode == 0

    # Each line should be valid JSON
    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    # First event should be "start"
    assert events[0]["type"] == "start"
    assert "source" in events[0]

    # Last event should be "end" with both timestamp formats by default
    assert events[-1]["type"] == "end"
    assert "total_time_ms" in events[-1]
    assert isinstance(events[-1]["total_time_ms"], int)
    assert "total_time_formatted" in events[-1]
    assert isinstance(events[-1]["total_time_formatted"], str)

    # Middle events should be "pattern_detected" with both timestamp formats by default
    for event in events[1:-1]:
        assert event["type"] == "pattern_detected"
        assert "clip_name" in event
        assert "timestamp_ms" in event
        assert isinstance(event["timestamp_ms"], int)
        assert "timestamp_formatted" in event
        assert isinstance(event["timestamp_formatted"], str)


def test_match_jsonl_timestamp_format_ms():
    """Test --timestamp-format ms produces integer milliseconds only."""
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
        "--timestamp-format", "ms",
    )
    assert result.returncode == 0

    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    assert "total_time_ms" in events[-1]
    assert isinstance(events[-1]["total_time_ms"], int)
    assert "total_time_formatted" not in events[-1]

    for event in events[1:-1]:
        assert event["type"] == "pattern_detected"
        assert "timestamp_ms" in event
        assert isinstance(event["timestamp_ms"], int)
        assert "timestamp_formatted" not in event


def test_match_jsonl_timestamp_format_formatted():
    """Test --timestamp-format formatted produces HH:MM:SS.mmm strings."""
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
        "--timestamp-format", "formatted",
    )
    assert result.returncode == 0

    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    # End event should have formatted total_time
    assert "total_time_formatted" in events[-1]
    assert isinstance(events[-1]["total_time_formatted"], str)
    assert "total_time_ms" not in events[-1]

    # Pattern events should have formatted timestamps
    for event in events[1:-1]:
        assert event["type"] == "pattern_detected"
        assert "timestamp_formatted" in event
        assert isinstance(event["timestamp_formatted"], str)
        assert "timestamp_ms" not in event


def test_match_jsonl_start_event_source():
    """Test start event contains correct source."""
    # Test with audio file
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
    )
    start_event = json.loads(result.stdout.strip().split("\n")[0])
    assert "rthk_section_with_beep.wav" in start_event["source"]

    # Test with stdin (WAV mode, always outputs JSONL)
    wav_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav")

    result = run_cli_binary(
        "match",
        "--stdin",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
        stdin_data=wav_data,
    )
    start_event = json.loads(result.stdout.decode().strip().split("\n")[0])
    assert start_event["source"] == "stdin"


def test_match_jsonl_no_match_only_start_end():
    """Test JSONL output when no patterns match."""
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/cbs_news.wav",
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
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
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
    assert "rthk_beep" in config["clips"]


def test_show_config_clip_info():
    """Test show-config returns per-clip configuration."""
    result = run_cli(
        "show-config",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
    )
    config = json.loads(result.stdout)

    clip_config = config["clips"]["rthk_beep"]
    assert "duration_seconds" in clip_config
    assert "sliding_window_seconds" in clip_config


# --- Convert Command Tests ---


# --- Error Handling Tests ---


def test_match_nonexistent_audio_file():
    """Test match with nonexistent audio file."""
    result = run_cli(
        "match",
        "nonexistent.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
        check=False,
    )
    assert result.returncode != 0


def test_match_nonexistent_pattern_file():
    """Test match with nonexistent pattern file."""
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "nonexistent.wav",
        check=False,
    )
    assert result.returncode != 0


def test_match_no_audio_source():
    """Test match without any audio source."""
    result = run_cli(
        "match",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
        check=False,
    )
    assert result.returncode != 0
    assert "Please provide" in result.stderr


def test_match_no_pattern():
    """Test match without any pattern."""
    result = run_cli(
        "match",
        "sample_audios/rthk_section_with_beep.wav",
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
        "sample_audios/test_16khz/rthk_section_with_beep_16k.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
    )
    assert result.returncode == 0

    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should have pattern_detected events despite sample rate difference
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) > 0
    assert pattern_events[0]["clip_name"] == "rthk_beep"


# --- Stdin with Sample Rate Tests ---


def test_stdin_wav_with_wrong_sample_rate_rejected():
    """Test --stdin WAV mode rejects audio with wrong sample rate."""
    # Generate WAV at 16kHz (default target is 8kHz)
    wav_data = _convert_file_to_wav_stdout("sample_audios/rthk_section_with_beep.wav", sample_rate=16000)

    result = run_cli_binary(
        "match",
        "--stdin",
        "--pattern-file", "sample_audios/clips/rthk_beep.apd.toml",
        stdin_data=wav_data,
        check=False,
    )
    assert result.returncode != 0
    assert b"Expected 8000 Hz" in result.stderr


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
    # The multiplexed-stdin protocol is WAV-only, so this uses the cbs_news
    # .wav pattern (not the .apd.toml pure-tone pattern).
    with open("sample_audios/clips/cbs_news.wav", "rb") as f:
        pattern_data = f.read()

    # Convert audio to WAV
    audio_data = _convert_file_to_wav_stdout("sample_audios/cbs_news_audio_section.wav")

    # Build multiplexed payload
    payload = _build_multiplexed_payload(
        patterns=[("cbs_news", pattern_data)],
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
    assert pattern_events[0]["clip_name"] == "cbs_news"


def test_multiplexed_stdin_multiple_patterns():
    """Test --multiplexed-stdin with multiple patterns."""
    # The multiplexed-stdin protocol is WAV-only, so both patterns are .wav.
    with open("sample_audios/clips/cbs_news.wav", "rb") as f:
        pattern1_data = f.read()
    with open("sample_audios/clips/天空下的彩虹intro.wav", "rb") as f:
        pattern2_data = f.read()

    # Match against CBS audio: should detect cbs_news, not the rainbow intro.
    audio_data = _convert_file_to_wav_stdout("sample_audios/cbs_news_audio_section.wav")

    payload = _build_multiplexed_payload(
        patterns=[
            ("cbs_news", pattern1_data),
            ("rainbow_intro", pattern2_data),
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

    # Should detect cbs_news but not rainbow_intro
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    clip_names = {e["clip_name"] for e in pattern_events}
    assert "cbs_news" in clip_names


def test_multiplexed_stdin_requires_no_pattern_file():
    """Test --multiplexed-stdin does not require --pattern-file."""
    # Load pattern WAV (multiplexed-stdin protocol is WAV-only).
    with open("sample_audios/clips/cbs_news.wav", "rb") as f:
        pattern_data = f.read()

    # Convert audio to WAV
    audio_data = _convert_file_to_wav_stdout("sample_audios/cbs_news_audio_section.wav")

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

