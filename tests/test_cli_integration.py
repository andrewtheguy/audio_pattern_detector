"""CLI integration tests for audio-pattern-detector.

Tests the command-line interface for all modes and major options (except --debug).
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


def test_cli_match_help():
    """Test that match --help shows all options."""
    result = run_cli("match", "--help")
    assert result.returncode == 0
    assert "--pattern-file" in result.stdout
    assert "--pattern-folder" in result.stdout
    assert "--audio-file" in result.stdout
    assert "--audio-folder" in result.stdout
    assert "--audio-url" in result.stdout
    assert "--stdin" in result.stdout
    assert "--input-format" in result.stdout
    assert "--jsonl" in result.stdout


def test_cli_convert_help():
    """Test that convert --help shows options."""
    result = run_cli("convert", "--help")
    assert result.returncode == 0
    assert "--audio-file" in result.stdout
    assert "--dest-file" in result.stdout


def test_cli_no_command():
    """Test CLI with no command shows help."""
    result = run_cli(check=False)
    assert result.returncode == 1


# --- Match Command: --audio-file Tests ---


def test_match_audio_file_single_pattern():
    """Test match with --audio-file and single --pattern-file."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
    )
    assert result.returncode == 0

    # Parse JSON output
    output = json.loads(result.stdout)
    assert "rthk_beep" in output
    assert len(output["rthk_beep"]) == 2

    # Verify timestamps
    expected_times = [1.4165, 2.419125]
    for actual, expected in zip(sorted(output["rthk_beep"]), expected_times):
        assert abs(actual - expected) < 0.01


def test_match_audio_file_no_match():
    """Test match with audio file where pattern doesn't exist."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/cbs_news.wav",
    )
    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert "cbs_news" in output
    assert len(output["cbs_news"]) == 0


def test_match_audio_file_cbs_news():
    """Test match with CBS news audio and pattern."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/cbs_news_audio_section.wav",
        "--pattern-file", "sample_audios/clips/cbs_news.wav",
    )
    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert "cbs_news" in output
    assert len(output["cbs_news"]) == 1
    assert abs(output["cbs_news"][0] - 25.89875) < 0.01


def test_match_audio_file_nonexistent():
    """Test match with nonexistent audio file."""
    result = run_cli(
        "match",
        "--audio-file", "nonexistent.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        check=False,
    )
    assert result.returncode != 0


def test_match_pattern_file_nonexistent():
    """Test match with nonexistent pattern file."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "nonexistent.wav",
        check=False,
    )
    assert result.returncode != 0


# --- Match Command: --pattern-folder Tests ---


def test_match_pattern_folder():
    """Test match with --pattern-folder containing multiple patterns."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/cbs_news_audio_section.wav",
        "--pattern-folder", "sample_audios/clips",
    )
    assert result.returncode == 0

    output = json.loads(result.stdout)

    # Should have results for multiple patterns
    assert "cbs_news" in output
    assert "cbs_news_dada" in output
    assert "rthk_beep" in output

    # CBS patterns should match
    assert len(output["cbs_news"]) == 1
    assert len(output["cbs_news_dada"]) == 1
    # RTHK beep should not match CBS audio
    assert len(output["rthk_beep"]) == 0


def test_match_pattern_folder_rthk():
    """Test match with pattern folder against RTHK audio."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-folder", "sample_audios/clips",
    )
    assert result.returncode == 0

    output = json.loads(result.stdout)

    # RTHK beep should match
    assert len(output["rthk_beep"]) == 2
    # CBS patterns should not match
    assert len(output["cbs_news"]) == 0
    assert len(output["cbs_news_dada"]) == 0


# --- Match Command: --stdin Tests ---


def test_match_stdin_wav():
    """Test match reading audio from stdin."""
    # Read audio file as binary
    with open("sample_audios/rthk_section_with_beep.wav", "rb") as f:
        audio_data = f.read()

    result = run_cli_binary(
        "match",
        "--stdin",
        "--input-format", "wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        stdin_data=audio_data,
    )
    assert result.returncode == 0

    output = json.loads(result.stdout.decode())
    assert "rthk_beep" in output
    assert len(output["rthk_beep"]) == 2


def test_match_stdin_no_match():
    """Test match from stdin where pattern doesn't exist."""
    with open("sample_audios/rthk_section_with_beep.wav", "rb") as f:
        audio_data = f.read()

    result = run_cli_binary(
        "match",
        "--stdin",
        "--input-format", "wav",
        "--pattern-file", "sample_audios/clips/cbs_news.wav",
        stdin_data=audio_data,
    )
    assert result.returncode == 0

    output = json.loads(result.stdout.decode())
    assert "cbs_news" in output
    assert len(output["cbs_news"]) == 0


def test_match_stdin_cbs_news():
    """Test match from stdin with CBS news audio."""
    with open("sample_audios/cbs_news_audio_section.wav", "rb") as f:
        audio_data = f.read()

    result = run_cli_binary(
        "match",
        "--stdin",
        "--input-format", "wav",
        "--pattern-file", "sample_audios/clips/cbs_news.wav",
        stdin_data=audio_data,
    )
    assert result.returncode == 0

    output = json.loads(result.stdout.decode())
    assert "cbs_news" in output
    assert len(output["cbs_news"]) == 1


def test_match_stdin_with_pattern_folder():
    """Test match from stdin with pattern folder."""
    with open("sample_audios/cbs_news_audio_section.wav", "rb") as f:
        audio_data = f.read()

    result = run_cli_binary(
        "match",
        "--stdin",
        "--input-format", "wav",
        "--pattern-folder", "sample_audios/clips",
        stdin_data=audio_data,
    )
    assert result.returncode == 0

    output = json.loads(result.stdout.decode())
    assert "cbs_news" in output
    assert len(output["cbs_news"]) == 1


# --- Match Command: --jsonl Tests ---


def test_match_jsonl_output():
    """Test match with --jsonl flag produces JSONL output."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--jsonl",
    )
    assert result.returncode == 0

    # Parse JSONL output (one JSON object per line)
    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should have start, pattern_detected events, and end
    event_types = [e["type"] for e in events]
    assert event_types[0] == "start"
    assert event_types[-1] == "end"

    # Should have pattern_detected events
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) == 2

    # Verify pattern_detected event structure
    for event in pattern_events:
        assert "clip_name" in event
        assert "timestamp" in event
        assert "timestamp_formatted" in event
        assert event["clip_name"] == "rthk_beep"


def test_match_jsonl_start_event():
    """Test that --jsonl start event contains source info."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--jsonl",
    )

    lines = result.stdout.strip().split("\n")
    start_event = json.loads(lines[0])

    assert start_event["type"] == "start"
    assert "source" in start_event
    assert "rthk_section_with_beep.wav" in start_event["source"]


def test_match_jsonl_end_event():
    """Test that --jsonl end event contains timing info."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--jsonl",
    )

    lines = result.stdout.strip().split("\n")
    end_event = json.loads(lines[-1])

    assert end_event["type"] == "end"
    assert "total_time" in end_event
    assert "total_time_formatted" in end_event
    assert end_event["total_time"] > 0


def test_match_jsonl_no_match():
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

    # Should only have start and end events
    assert len(events) == 2
    assert events[0]["type"] == "start"
    assert events[1]["type"] == "end"


def test_match_jsonl_with_stdin():
    """Test --jsonl with --stdin input."""
    with open("sample_audios/rthk_section_with_beep.wav", "rb") as f:
        audio_data = f.read()

    result = run_cli_binary(
        "match",
        "--stdin",
        "--input-format", "wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--jsonl",
        stdin_data=audio_data,
    )
    assert result.returncode == 0

    lines = result.stdout.decode().strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Start event should indicate stdin source
    assert events[0]["type"] == "start"
    assert events[0]["source"] == "stdin"

    # Should have pattern_detected events
    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) == 2


def test_match_jsonl_multiple_patterns():
    """Test --jsonl with multiple patterns."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/cbs_news_audio_section.wav",
        "--pattern-folder", "sample_audios/clips",
        "--jsonl",
    )
    assert result.returncode == 0

    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    pattern_events = [e for e in events if e["type"] == "pattern_detected"]

    # Should have events for cbs_news and cbs_news_dada (not rthk_beep)
    clip_names = {e["clip_name"] for e in pattern_events}
    assert "cbs_news" in clip_names
    assert "cbs_news_dada" in clip_names


def test_match_jsonl_events_are_flushed():
    """Test that --jsonl events are flushed immediately (implicit test).

    We test this by ensuring events are properly ordered and complete.
    """
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/rthk_section_with_beep.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--jsonl",
    )

    lines = result.stdout.strip().split("\n")

    # All lines should be valid JSON
    for line in lines:
        json.loads(line)  # Should not raise

    # First event should be start, last should be end
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    assert first["type"] == "start"
    assert last["type"] == "end"


# --- Convert Command Tests ---


def test_convert_audio_file():
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


def test_convert_and_use_pattern():
    """Test converting a pattern and using it for matching."""
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
        assert len(output[pattern_name]) == 2
    finally:
        if Path(converted_pattern).exists():
            os.unlink(converted_pattern)


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


# --- Error Handling Tests ---


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


# --- 16kHz Audio Tests ---


def test_match_16khz_audio():
    """Test match with 16kHz audio file (auto-converted to 8kHz)."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/test_16khz/rthk_section_with_beep_16k.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
    )
    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert "rthk_beep" in output
    assert len(output["rthk_beep"]) == 2

    # Timestamps should be similar to 8kHz version
    expected_times = [1.4165, 2.419125]
    for actual, expected in zip(sorted(output["rthk_beep"]), expected_times):
        assert abs(actual - expected) < 0.05  # Slightly larger tolerance for resampling


def test_match_16khz_audio_jsonl():
    """Test match with 16kHz audio in JSONL mode."""
    result = run_cli(
        "match",
        "--audio-file", "sample_audios/test_16khz/rthk_section_with_beep_16k.wav",
        "--pattern-file", "sample_audios/clips/rthk_beep.wav",
        "--jsonl",
    )
    assert result.returncode == 0

    lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in lines]

    pattern_events = [e for e in events if e["type"] == "pattern_detected"]
    assert len(pattern_events) == 2
