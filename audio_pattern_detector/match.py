import argparse
import glob
import json
import os
import struct
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import PatternDetectedCallback
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from andrew_utils import seconds_to_time
from audio_pattern_detector.audio_utils import (
    ffmpeg_get_float32_pcm,
    resample_audio,
    DEFAULT_TARGET_SAMPLE_RATE,
)

def _emit_jsonl(event_type: str, **kwargs: Any) -> None:
    """Emit a JSONL event to stdout and flush immediately."""
    event = {"type": event_type, **kwargs}
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _read_uint32(stream: Any) -> int:
    """Read a uint32 (little-endian) from a binary stream."""
    data = stream.read(4)
    if len(data) < 4:
        raise ValueError(f"Unexpected EOF reading uint32 (got {len(data)} bytes)")
    return int.from_bytes(data, byteorder='little', signed=False)


def _read_patterns_from_multiplexed_stdin(
    target_sample_rate: int,
) -> list[AudioClip]:
    """Read pattern clips from multiplexed stdin protocol.

    Protocol format (all integers are uint32 little-endian):
        [4 bytes] number_of_patterns
        For each pattern:
            [4 bytes] name_length
            [name_length bytes] name (UTF-8)
            [4 bytes] data_length
            [data_length bytes] WAV data

    After patterns are read, the remaining stdin contains audio data.

    Args:
        target_sample_rate: Target sample rate for pattern clips.

    Returns:
        List of AudioClip instances loaded from stdin.
    """
    stdin = sys.stdin.buffer

    num_patterns = _read_uint32(stdin)
    if num_patterns == 0:
        raise ValueError("No patterns provided in multiplexed stdin")
    if num_patterns > 100:
        raise ValueError(f"Too many patterns ({num_patterns}), max is 100")

    print(f"Reading {num_patterns} pattern(s) from multiplexed stdin...", file=sys.stderr)

    pattern_clips: list[AudioClip] = []
    for i in range(num_patterns):
        # Read pattern name
        name_length = _read_uint32(stdin)
        if name_length == 0 or name_length > 1024:
            raise ValueError(f"Invalid pattern name length: {name_length}")
        name_bytes = stdin.read(name_length)
        if len(name_bytes) < name_length:
            raise ValueError(f"Unexpected EOF reading pattern name {i+1}")
        name = name_bytes.decode('utf-8')

        # Read pattern WAV data
        data_length = _read_uint32(stdin)
        if data_length == 0:
            raise ValueError(f"Pattern '{name}' has zero-length data")
        if data_length > 100 * 1024 * 1024:  # 100MB max per pattern
            raise ValueError(f"Pattern '{name}' data too large: {data_length} bytes")
        wav_data = stdin.read(data_length)
        if len(wav_data) < data_length:
            raise ValueError(f"Unexpected EOF reading pattern '{name}' data")

        # Create AudioClip from WAV bytes
        clip = AudioClip.from_wav_bytes(wav_data, name, sample_rate=target_sample_rate)
        pattern_clips.append(clip)
        print(f"  Loaded pattern '{name}' ({clip.clip_length_seconds():.2f}s)", file=sys.stderr)

    return pattern_clips


def match_pattern(
    audio_source: str | None,
    pattern_files: list[str],
    debug_mode: bool = False,
    on_pattern_detected: PatternDetectedCallback | None = None,
    accumulate_results: bool = True,
    seconds_per_chunk: int | None = 60,
    from_stdin: bool = False,
    target_sample_rate: int | None = None,
    debug_dir: str = './tmp',
    height_min: float | None = None,
) -> tuple[dict[str, list[float]] | None, float]:
    """Find pattern matches in audio file or stdin

    Args:
        audio_source: Path to audio file, or None if from_stdin=True
        pattern_files: List of pattern file paths
        debug_mode: Enable debug mode
        on_pattern_detected: Optional callback for streaming output.
                             Signature: on_pattern_detected(clip_name: str, timestamp: float)
        accumulate_results: If False, don't accumulate results (saves memory for streaming)
        seconds_per_chunk: Seconds per chunk for sliding window (None for auto-compute)
        from_stdin: Whether to read from stdin (WAV format only)
        target_sample_rate: Target sample rate for processing (default: DEFAULT_TARGET_SAMPLE_RATE, 8000)
        height_min: Override minimum correlation peak height (default: 0.25).
    """
    if not from_stdin:
        if audio_source is None or not os.path.exists(audio_source):
            raise ValueError(f"Audio {audio_source} does not exist")

    # Use DEFAULT_TARGET_SAMPLE_RATE as default if not specified
    sr = target_sample_rate if target_sample_rate is not None else DEFAULT_TARGET_SAMPLE_RATE

    pattern_clips = []
    clip_names_seen: dict[str, str] = {}  # name -> file path for error messages
    for pattern_file in pattern_files:
        if not os.path.exists(pattern_file):
            raise ValueError(f"Pattern {pattern_file} does not exist")
        pattern_clip = AudioClip.from_audio_file(pattern_file, sample_rate=sr)
        # Check for duplicate clip names
        if pattern_clip.name in clip_names_seen:
            raise ValueError(
                f"Duplicate clip name '{pattern_clip.name}' from files:\n"
                f"  - {clip_names_seen[pattern_clip.name]}\n"
                f"  - {pattern_file}\n"
                f"Use --pattern-file with name=path syntax to specify unique names."
            )
        clip_names_seen[pattern_clip.name] = pattern_file
        pattern_clips.append(pattern_clip)

    if len(pattern_clips) == 0:
        raise ValueError("No pattern clips passed")

    if from_stdin:
        return _match_pattern_wav_stdin(
            pattern_clips=pattern_clips,
            debug_mode=debug_mode,
            on_pattern_detected=on_pattern_detected,
            accumulate_results=accumulate_results,
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=sr,
            debug_dir=debug_dir,
            height_min=height_min,
        )

    # File mode - audio_source is guaranteed to be str here since from_stdin=False
    assert audio_source is not None
    audio_name = Path(audio_source).stem
    print(f"Finding pattern in audio file {audio_name}...", file=sys.stderr)

    if audio_source.lower().endswith('.wav'):
        stream_wrapper = _WavFileStreamWrapper(audio_source, sr)
        try:
            full_streaming_audio = AudioStream(name=audio_name, audio_stream=stream_wrapper, sample_rate=sr)
            peak_times, total_time = (
                AudioPatternDetector(
                    debug_mode=debug_mode,
                    audio_clips=pattern_clips,
                    seconds_per_chunk=seconds_per_chunk,
                    target_sample_rate=sr,
                    debug_dir=debug_dir,
                    height_min=height_min,
                )
                .find_clip_in_audio(
                    full_streaming_audio,
                    on_pattern_detected=on_pattern_detected,
                    accumulate_results=accumulate_results,
                )
            )
        finally:
            stream_wrapper.close()
        return peak_times, total_time

    with ffmpeg_get_float32_pcm(
        audio_source,
        target_sample_rate=sr,
        ac=1,
    ) as stdout:
        full_streaming_audio = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)
        # Find clip occurrences in the full audio
        peak_times, total_time = (
            AudioPatternDetector(
                debug_mode=debug_mode,
                audio_clips=pattern_clips,
                seconds_per_chunk=seconds_per_chunk,
                target_sample_rate=sr,
                debug_dir=debug_dir,
            )
            .find_clip_in_audio(
                full_streaming_audio,
                on_pattern_detected=on_pattern_detected,
                accumulate_results=accumulate_results,
            )
        )
    return peak_times, total_time


def _validate_wav_header(stream: Any, target_sample_rate: int) -> tuple[int, int]:
    """Read and validate a WAV header from a binary stream.

    Accepts mono WAV at target sample rate in 16-bit PCM, 32-bit PCM,
    or 32-bit IEEE float. Returns (audio_format, bits_per_sample).
    Raises ValueError on invalid format.
    """
    riff = stream.read(4)
    if riff != b'RIFF':
        raise ValueError(f"Not a WAV file: expected RIFF, got {riff!r}")

    stream.read(4)  # file size (ignore)

    wave_sig = stream.read(4)
    if wave_sig != b'WAVE':
        raise ValueError(f"Not a WAV file: expected WAVE, got {wave_sig!r}")

    # Find fmt chunk
    while True:
        chunk_id = stream.read(4)
        if len(chunk_id) < 4:
            raise ValueError("WAV file missing fmt chunk")
        chunk_size = struct.unpack('<I', stream.read(4))[0]

        if chunk_id == b'fmt ':
            break
        skipped = stream.read(chunk_size)
        if len(skipped) != chunk_size:
            raise ValueError("WAV file truncated while skipping chunk")

    fmt_data = stream.read(chunk_size)
    if len(fmt_data) < 16:
        raise ValueError("WAV fmt chunk too short")

    audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack(
        '<HHIIHH', fmt_data[:16]
    )

    if audio_format == 1:  # PCM integer
        if bits_per_sample not in (16, 32):
            raise ValueError(f"Expected 16-bit or 32-bit PCM, got {bits_per_sample}")
    elif audio_format == 3:  # IEEE float
        if bits_per_sample != 32:
            raise ValueError(f"Expected 32-bit float, got {bits_per_sample}")
    else:
        raise ValueError(f"Expected PCM (1) or IEEE float (3) format, got {audio_format}")

    if channels != 1:
        raise ValueError(f"Expected mono (1 channel), got {channels}")
    if sample_rate != target_sample_rate:
        raise ValueError(f"Expected {target_sample_rate} Hz, got {sample_rate}")

    # Find data chunk
    while True:
        chunk_id = stream.read(4)
        if len(chunk_id) < 4:
            raise ValueError("WAV file missing data chunk")
        chunk_size_bytes = stream.read(4)
        if len(chunk_size_bytes) < 4:
            raise ValueError("WAV file truncated")

        if chunk_id == b'data':
            break
        chunk_size = struct.unpack('<I', chunk_size_bytes)[0]
        skipped = stream.read(chunk_size)
        if len(skipped) != chunk_size:
            raise ValueError("WAV file truncated while skipping chunk")

    return audio_format, bits_per_sample


class _WavStdinStreamWrapper:
    """Wrapper to read WAV format from stdin.

    Parses WAV header manually to detect audio format (PCM vs IEEE float).
    Requires mono audio at target sample rate.
    Passes float32 data through without extra conversion.
    """

    def __init__(self, target_sample_rate: int) -> None:
        self._audio_format, self._bits_per_sample = _validate_wav_header(
            sys.stdin.buffer, target_sample_rate,
        )

        if self._audio_format == 3:
            self._dtype = np.dtype(np.float32)
        elif self._bits_per_sample == 16:
            self._dtype = np.dtype(np.int16)
        else:
            self._dtype = np.dtype(np.int32)

        fmt_name = "float32" if self._audio_format == 3 else f"int{self._bits_per_sample}"
        print(f"WAV stdin: {target_sample_rate}Hz, mono, {fmt_name}", file=sys.stderr)

    def read(self, size: int, /) -> bytes:
        """Read and convert audio data from WAV stdin.

        Args:
            size: Number of bytes to return (float32)

        Returns:
            Raw bytes of float32 PCM
        """
        # Adjust read size for input dtype
        target_samples = size // 4  # output is float32, 4 bytes per sample
        read_bytes = target_samples * self._dtype.itemsize

        data = sys.stdin.buffer.read(read_bytes)
        if not data:
            return b""

        raw = np.frombuffer(data, dtype=self._dtype)
        if self._dtype == np.int16:
            return (raw.astype(np.float32) / np.float32(32768.0)).tobytes()
        elif self._dtype == np.int32:
            return (raw.astype(np.float32) / np.float32(2147483648.0)).tobytes()
        else:
            return raw.tobytes()


class _WavFileStreamWrapper:
    """Wrapper to read WAV format from a file.

    Reads WAV header to get sample rate, then streams audio data.
    Automatically converts to target sample rate if needed.
    """

    def __init__(self, file_path: str, target_sample_rate: int) -> None:
        self.target_sample_rate = target_sample_rate
        self._bytes_per_sample = 4  # output is float32
        self._validated = False
        self._file_path = file_path

        # Read WAV header from file
        try:
            self._wav: wave.Wave_read = wave.open(file_path, 'rb')
        except (wave.Error, FileNotFoundError, OSError) as e:
            raise ValueError(f"Failed to read WAV file {file_path}: {e}")

        self.input_sample_rate = self._wav.getframerate()
        self._channels = self._wav.getnchannels()
        self._sampwidth = self._wav.getsampwidth()
        self.needs_resample = self.input_sample_rate != target_sample_rate

        if self._channels != 1:
            print(f"Warning: WAV has {self._channels} channels, will be mixed to mono", file=sys.stderr)

    def _validate_first_chunk(self, audio: NDArray[np.float32]) -> None:
        """Check first chunk for signs of corrupt audio."""
        if self._validated or len(audio) == 0:
            return
        self._validated = True

        warnings: list[str] = []
        if np.any(np.isnan(audio)):
            warnings.append("Audio contains NaN values - data may be corrupt")
        if np.any(np.isinf(audio)):
            warnings.append("Audio contains Inf values - data may be corrupt")

        max_abs = np.max(np.abs(audio))
        if max_abs > 1.5:
            warnings.append(f"Audio values exceed expected range (max: {max_abs:.2f})")

        if np.all(audio == 0):
            warnings.append("First chunk is all zeros - verify input is correct")

        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)

    def read(self, size: int, /) -> bytes:
        """Read and convert audio data from WAV file.

        Args:
            size: Number of bytes to return (at target sample rate, float32)

        Returns:
            Raw bytes of float32 PCM at target sample rate
        """
        # Calculate how many frames to read
        target_samples = size // self._bytes_per_sample
        if self.needs_resample:
            input_samples = int(target_samples * self.input_sample_rate / self.target_sample_rate)
        else:
            input_samples = target_samples

        # Read raw frames from WAV
        raw_data = self._wav.readframes(input_samples)
        if not raw_data:
            return b""

        # Convert to float32 based on sample width
        if self._sampwidth == 2:  # 16-bit
            audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif self._sampwidth == 4:  # 32-bit
            audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif self._sampwidth == 1:  # 8-bit unsigned
            audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {self._sampwidth} bytes")

        # Mix to mono if stereo
        if self._channels > 1:
            audio = audio.reshape(-1, self._channels).mean(axis=1).astype(np.float32)

        # Validate first chunk
        if not self._validated:
            self._validate_first_chunk(audio)

        # Resample if needed
        if self.needs_resample:
            audio = resample_audio(audio, self.input_sample_rate, self.target_sample_rate)

        return audio.tobytes()

    def close(self) -> None:
        """Close the WAV file."""
        self._wav.close()


def _match_pattern_wav_stdin(
    pattern_clips: list[AudioClip],
    debug_mode: bool,
    on_pattern_detected: PatternDetectedCallback | None,
    accumulate_results: bool,
    seconds_per_chunk: int | None,
    target_sample_rate: int,
    debug_dir: str = './tmp',
    height_min: float | None = None,
) -> tuple[dict[str, list[float]] | None, float]:
    """Internal function to handle WAV stdin mode."""
    # Create stream wrapper that reads WAV header and handles resampling
    stream_wrapper = _WavStdinStreamWrapper(target_sample_rate)

    audio_name = "stdin"
    print(f"Finding pattern in audio stream {audio_name}...", file=sys.stderr)

    full_streaming_audio = AudioStream(
        name=audio_name,
        audio_stream=stream_wrapper,
        sample_rate=target_sample_rate,
    )

    # Find clip occurrences in the full audio
    peak_times, total_time = (
        AudioPatternDetector(
            debug_mode=debug_mode,
            audio_clips=pattern_clips,
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=target_sample_rate,
            debug_dir=debug_dir,
            height_min=height_min,
        )
        .find_clip_in_audio(
            full_streaming_audio,
            on_pattern_detected=on_pattern_detected,
            accumulate_results=accumulate_results,
        )
    )

    return peak_times, total_time


def _match_pattern_multiplexed_stdin(
    debug_mode: bool,
    on_pattern_detected: PatternDetectedCallback | None,
    accumulate_results: bool,
    seconds_per_chunk: int | None,
    target_sample_rate: int,
    debug_dir: str = './tmp',
    height_min: float | None = None,
) -> tuple[dict[str, list[float]] | None, float]:
    """Internal function to handle multiplexed stdin mode.

    Reads patterns from stdin first using the multiplexed protocol,
    then reads WAV audio stream from remaining stdin.
    """
    # Read patterns from multiplexed protocol
    pattern_clips = _read_patterns_from_multiplexed_stdin(target_sample_rate)

    print("Reading WAV audio from stdin...", file=sys.stderr)
    stream_wrapper = _WavStdinStreamWrapper(target_sample_rate)

    audio_name = "stdin"
    full_streaming_audio = AudioStream(
        name=audio_name,
        audio_stream=stream_wrapper,
        sample_rate=target_sample_rate,
    )

    # Find clip occurrences in the full audio
    peak_times, total_time = (
        AudioPatternDetector(
            debug_mode=debug_mode,
            audio_clips=pattern_clips,
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=target_sample_rate,
            debug_dir=debug_dir,
            height_min=height_min,
        )
        .find_clip_in_audio(
            full_streaming_audio,
            on_pattern_detected=on_pattern_detected,
            accumulate_results=accumulate_results,
        )
    )

    return peak_times, total_time


def _make_jsonl_callback(timestamp_format: str = "both") -> PatternDetectedCallback:
    """Create a callback that emits pattern_detected JSONL events."""
    last_ms: dict[str, int] = {}
    def callback(clip_name: str, timestamp: float) -> None:
        ts_ms = round(timestamp * 1000)
        if last_ms.get(clip_name) == ts_ms:
            return
        last_ms[clip_name] = ts_ms
        if timestamp_format == "formatted":
            _emit_jsonl(
                "pattern_detected",
                clip_name=clip_name,
                timestamp_formatted=seconds_to_time(timestamp),
            )
        elif timestamp_format == "ms":
            _emit_jsonl(
                "pattern_detected",
                clip_name=clip_name,
                timestamp_ms=ts_ms,
            )
        else:
            _emit_jsonl(
                "pattern_detected",
                clip_name=clip_name,
                timestamp_ms=ts_ms,
                timestamp_formatted=seconds_to_time(timestamp),
            )
    return callback


def _emit_jsonl_end(total_time: float, timestamp_format: str = "both") -> None:
    """Emit a JSONL end event with the appropriate timestamp format."""
    if timestamp_format == "formatted":
        _emit_jsonl("end", total_time_formatted=seconds_to_time(total_time))
    elif timestamp_format == "ms":
        _emit_jsonl("end", total_time_ms=round(total_time * 1000))
    else:
        _emit_jsonl(
            "end",
            total_time_ms=round(total_time * 1000),
            total_time_formatted=seconds_to_time(total_time),
        )


def _run_match_with_output(
    args: argparse.Namespace,
    pattern_files: list[str],
    audio_source: str | None,
    from_stdin: bool = False,
    seconds_per_chunk: int | None = 60,
    target_sample_rate: int | None = None,
    debug_dir: str = './tmp',
    height_min: float | None = None,
) -> tuple[None, float]:
    """Run match_pattern and handle JSONL streaming output."""
    timestamp_format: str = getattr(args, 'timestamp_format', 'both')

    callback = _make_jsonl_callback(timestamp_format)
    _emit_jsonl("start", source="stdin" if from_stdin else (audio_source or "unknown"))

    _, total_time = match_pattern(
        audio_source,
        pattern_files,
        debug_mode=args.debug,
        on_pattern_detected=callback,
        accumulate_results=False,
        seconds_per_chunk=seconds_per_chunk,
        from_stdin=from_stdin,
        target_sample_rate=target_sample_rate,
        debug_dir=debug_dir,
        height_min=height_min,
    )
    print(f"Total time processed: {seconds_to_time(seconds=total_time)}", file=sys.stderr)

    _emit_jsonl_end(total_time, timestamp_format)

    return None, total_time


def cmd_match(args: argparse.Namespace) -> None:
    """Handler for match subcommand"""
    # Parse chunk-seconds argument: "auto" -> None, otherwise int
    chunk_seconds_str = getattr(args, 'chunk_seconds', '60')
    if chunk_seconds_str.lower() == 'auto':
        seconds_per_chunk = None
    else:
        try:
            seconds_per_chunk = int(chunk_seconds_str)
        except ValueError:
            print(f"Error: --chunk-seconds must be 'auto' or a positive integer, got '{chunk_seconds_str}'", file=sys.stderr)
            sys.exit(1)

    # Get target sample rate (None means use default 8000)
    target_sample_rate = getattr(args, 'target_sample_rate', None)
    sr = target_sample_rate if target_sample_rate is not None else DEFAULT_TARGET_SAMPLE_RATE

    debug_dir: str = getattr(args, 'debug_dir', './tmp')
    height_min: float | None = getattr(args, 'height_min', None)

    # Handle multiplexed stdin mode (patterns + audio all from stdin)
    multiplexed_stdin = getattr(args, 'multiplexed_stdin', False)
    timestamp_format: str = getattr(args, 'timestamp_format', 'both')

    if multiplexed_stdin:
        # Multiplexed stdin mode: always JSONL output
        callback = _make_jsonl_callback(timestamp_format)
        _emit_jsonl("start", source="multiplexed-stdin")

        _, total_time = _match_pattern_multiplexed_stdin(
            debug_mode=args.debug,
            on_pattern_detected=callback,
            accumulate_results=False,
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=sr,
            debug_dir=debug_dir,
            height_min=height_min,
        )

        print(f"Total time processed: {seconds_to_time(seconds=total_time)}", file=sys.stderr)
        _emit_jsonl_end(total_time, timestamp_format)
        return

    # Non-multiplexed modes: require pattern file(s)
    pattern_files: list[str] = []
    if args.pattern_folder:
        for folder in args.pattern_folder:
            for ext in ("wav", "apd.toml"):
                for pattern_file in glob.glob(f'{folder}/*.{ext}'):
                    print(f"adding pattern file {pattern_file}...", file=sys.stderr)
                    pattern_files.append(pattern_file)
    if args.pattern_file:
        pattern_files.extend(args.pattern_file)

    if not pattern_files:
        print("Please provide either --pattern-file, --pattern-folder, or --multiplexed-stdin", file=sys.stderr)
        sys.exit(1)

    if args.stdin:
        _run_match_with_output(
            args, pattern_files, None,
            from_stdin=True,
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=target_sample_rate,
            debug_dir=debug_dir,
            height_min=height_min,
        )
    elif args.audio_file:
        _run_match_with_output(
            args, pattern_files, args.audio_file,
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=target_sample_rate,
            debug_dir=debug_dir,
            height_min=height_min,
        )
    else:
        print("Please provide an audio file or --stdin or --multiplexed-stdin", file=sys.stderr)
        sys.exit(1)


def cmd_show_config(args: argparse.Namespace) -> None:
    """Handler for show-config subcommand"""
    # Get target sample rate (None means use default 8000)
    target_sample_rate = getattr(args, 'target_sample_rate', None)

    pattern_file = args.pattern_file
    if not os.path.exists(pattern_file):
        print(f"Error: Pattern {pattern_file} does not exist", file=sys.stderr)
        sys.exit(1)
    pattern_clips = [AudioClip.from_audio_file(pattern_file, sample_rate=target_sample_rate)]

    # Use auto mode (None) to show minimum computed values
    detector = AudioPatternDetector(
        audio_clips=pattern_clips,
        debug_mode=False,
        seconds_per_chunk=None,
        target_sample_rate=target_sample_rate,
    )
    config = detector.get_config()
    print(json.dumps(config, indent=2, ensure_ascii=False))
