import argparse
import glob
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import PatternDetectedCallback
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import (
    ffmpeg_get_float32_pcm,
    resample_audio,
    seconds_to_time,
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
    raw_pcm: bool = False,
    sample_rate: int | None = None,
    target_sample_rate: int | None = None,
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
        from_stdin: Whether to read from stdin
        raw_pcm: If True and from_stdin=True, read raw float32 PCM (requires sample_rate).
                 If False and from_stdin=True, read WAV format from stdin.
        sample_rate: Sample rate of stdin input (required for raw_pcm mode, ignored for WAV mode)
        target_sample_rate: Target sample rate for processing (default: DEFAULT_TARGET_SAMPLE_RATE, 8000)
    """
    if not from_stdin:
        if audio_source is None or not os.path.exists(audio_source):
            raise ValueError(f"Audio {audio_source} does not exist")

    # Use DEFAULT_TARGET_SAMPLE_RATE as default if not specified
    sr = target_sample_rate if target_sample_rate is not None else DEFAULT_TARGET_SAMPLE_RATE

    pattern_clips = []
    for pattern_file in pattern_files:
        if not os.path.exists(pattern_file):
            raise ValueError(f"Pattern {pattern_file} does not exist")
        pattern_clip = AudioClip.from_audio_file(pattern_file, sample_rate=sr)
        pattern_clips.append(pattern_clip)

    if len(pattern_clips) == 0:
        raise ValueError("No pattern clips passed")

    if from_stdin:
        if raw_pcm:
            # Raw PCM mode: read raw float32 little-endian PCM directly
            input_sr = sample_rate if sample_rate is not None else sr
            return _match_pattern_raw_pcm(
                pattern_clips=pattern_clips,
                debug_mode=debug_mode,
                on_pattern_detected=on_pattern_detected,
                accumulate_results=accumulate_results,
                seconds_per_chunk=seconds_per_chunk,
                input_sample_rate=input_sr,
                target_sample_rate=sr,
            )
        else:
            # WAV mode: read WAV format from stdin
            return _match_pattern_wav_stdin(
                pattern_clips=pattern_clips,
                debug_mode=debug_mode,
                on_pattern_detected=on_pattern_detected,
                accumulate_results=accumulate_results,
                seconds_per_chunk=seconds_per_chunk,
                target_sample_rate=sr,
            )

    # File mode - audio_source is guaranteed to be str here since from_stdin=False
    assert audio_source is not None
    audio_name = Path(audio_source).stem
    print(f"Finding pattern in audio file {audio_name}...", file=sys.stderr)

    # For WAV files, use scipy (no ffmpeg needed)
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

    # For non-WAV files, use ffmpeg
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
            )
            .find_clip_in_audio(
                full_streaming_audio,
                on_pattern_detected=on_pattern_detected,
                accumulate_results=accumulate_results,
            )
        )
    return peak_times, total_time


class _WavStdinStreamWrapper:
    """Wrapper to read WAV format from stdin.

    Reads WAV header to get sample rate, then streams audio data.
    Automatically converts to target sample rate if needed.
    """

    def __init__(self, target_sample_rate: int) -> None:
        self.target_sample_rate = target_sample_rate
        self._bytes_per_sample = 4  # output is float32
        self._validated = False

        # Read WAV header from stdin
        try:
            self._wav: wave.Wave_read = wave.open(sys.stdin.buffer, 'rb')
        except wave.Error as e:
            raise ValueError(f"Failed to read WAV header from stdin: {e}. Use --raw-pcm for headerless PCM data.")

        self.input_sample_rate = self._wav.getframerate()
        self._channels = self._wav.getnchannels()
        self._sampwidth = self._wav.getsampwidth()
        self.needs_resample = self.input_sample_rate != target_sample_rate

        print(f"WAV stdin: {self.input_sample_rate}Hz, {self._channels} channel(s), {self._sampwidth*8}-bit", file=sys.stderr)

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
        """Read and convert audio data from WAV stdin.

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


class _WavFileStreamWrapper:
    """Wrapper to read WAV format from a file.

    Reads WAV header to get sample rate, then streams audio data.
    Automatically converts to target sample rate if needed.
    No ffmpeg required - uses Python's wave module and scipy for resampling.
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


class _RawPcmStreamWrapper:
    """Wrapper to make raw PCM stdin work like AudioStream expects.

    Reads raw float32 PCM from stdin, optionally resampling to target rate.
    Includes validation to detect potentially corrupt audio.
    """

    def __init__(self, input_sample_rate: int, target_sample_rate: int) -> None:
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.needs_resample = input_sample_rate != target_sample_rate
        self._bytes_per_sample = 4  # float32
        self._validated = False

    def _check_for_wav_header(self, data: bytes) -> None:
        """Check if data starts with WAV header and raise error if so.

        WAV file structure:
        - Bytes 0-3:   "RIFF" (0x52, 0x49, 0x46, 0x46)
        - Bytes 4-7:   File size minus 8
        - Bytes 8-11:  "WAVE" (0x57, 0x41, 0x56, 0x45)
        - Bytes 12-15: "fmt " (0x66, 0x6D, 0x74, 0x20) - format chunk ID

        If raw PCM mode receives WAV data, abort early with a helpful error message.
        """
        if len(data) < 16:
            return

        # Check for WAV header: RIFF....WAVEfmt
        is_riff = data[0:4] == b'RIFF'
        is_wave = data[8:12] == b'WAVE'
        is_fmt = data[12:16] == b'fmt '

        if is_riff and is_wave and is_fmt:
            raise ValueError(
                "Input appears to be WAV format (detected RIFF/WAVE/fmt header). "
                "Use --stdin without --raw-pcm for WAV input, or ensure input is raw float32 PCM."
            )

    def _validate_first_chunk(self, audio: NDArray[np.float32]) -> None:
        """Check first chunk for signs of corrupt audio and emit warnings."""
        if self._validated or len(audio) == 0:
            return
        self._validated = True

        warnings: list[str] = []

        # Check for NaN or Inf values (definite corruption)
        if np.any(np.isnan(audio)):
            warnings.append("Audio contains NaN values - data may be corrupt")
        if np.any(np.isinf(audio)):
            warnings.append("Audio contains Inf values - data may be corrupt")

        # Check value range - float32 PCM should be in [-1, 1]
        max_abs = np.max(np.abs(audio))
        if max_abs > 1.5:
            warnings.append(f"Audio values exceed expected range (max: {max_abs:.2f}) - may be wrong format or corrupt")

        # Check for excessive clipping (>10% of samples at ±1.0)
        clipped = np.sum(np.abs(audio) >= 0.9999)
        clip_ratio = clipped / len(audio)
        if clip_ratio > 0.1:
            warnings.append(f"Audio appears heavily clipped ({clip_ratio*100:.1f}% of samples)")

        # Check for DC offset (mean should be near 0)
        mean_val = np.mean(audio)
        if abs(mean_val) > 0.1:
            warnings.append(f"Audio has significant DC offset (mean: {mean_val:.3f})")

        # Check if audio is all zeros (silence or no data)
        if np.all(audio == 0):
            warnings.append("First chunk is all zeros - verify input is correct")

        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)

    def read(self, size: int, /) -> bytes:
        """Read and optionally resample audio data.

        Args:
            size: Number of bytes to return (at target sample rate)

        Returns:
            Raw bytes of float32 PCM at target sample rate
        """
        if not self.needs_resample:
            data = sys.stdin.buffer.read(size)
            if data and not self._validated:
                # Check for WAV header before processing as raw PCM
                self._check_for_wav_header(data)
                audio = np.frombuffer(data, dtype=np.float32)
                self._validate_first_chunk(audio)
            return data

        # Calculate how many input bytes we need to produce the requested output bytes
        target_samples = size // self._bytes_per_sample
        input_samples = int(target_samples * self.input_sample_rate / self.target_sample_rate)
        input_bytes = input_samples * self._bytes_per_sample

        # Read input data
        data = sys.stdin.buffer.read(input_bytes)
        if not data:
            return b""

        # Check for WAV header before processing as raw PCM
        if not self._validated:
            self._check_for_wav_header(data)

        # Convert to numpy, resample, and convert back to bytes
        audio = np.frombuffer(data, dtype=np.float32)
        if not self._validated:
            self._validate_first_chunk(audio)
        resampled = resample_audio(audio, self.input_sample_rate, self.target_sample_rate)
        return resampled.tobytes()


def _match_pattern_wav_stdin(
    pattern_clips: list[AudioClip],
    debug_mode: bool,
    on_pattern_detected: PatternDetectedCallback | None,
    accumulate_results: bool,
    seconds_per_chunk: int | None,
    target_sample_rate: int,
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
    raw_pcm: bool,
    input_sample_rate: int | None,
    target_sample_rate: int,
) -> tuple[dict[str, list[float]] | None, float]:
    """Internal function to handle multiplexed stdin mode.

    Reads patterns from stdin first using the multiplexed protocol,
    then reads audio stream from remaining stdin.
    """
    # Read patterns from multiplexed protocol
    pattern_clips = _read_patterns_from_multiplexed_stdin(target_sample_rate)

    # Now read audio from remaining stdin
    if raw_pcm:
        if input_sample_rate is None:
            raise ValueError("--source-sample-rate is required with --raw-pcm")
        print(f"Reading raw PCM audio from stdin (source sample rate: {input_sample_rate}Hz)...", file=sys.stderr)
        if input_sample_rate != target_sample_rate:
            print(f"Resampling from {input_sample_rate}Hz to {target_sample_rate}Hz...", file=sys.stderr)
        stream_wrapper = _RawPcmStreamWrapper(input_sample_rate, target_sample_rate)
    else:
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
        )
        .find_clip_in_audio(
            full_streaming_audio,
            on_pattern_detected=on_pattern_detected,
            accumulate_results=accumulate_results,
        )
    )

    return peak_times, total_time


def _match_pattern_raw_pcm(
    pattern_clips: list[AudioClip],
    debug_mode: bool,
    on_pattern_detected: PatternDetectedCallback | None,
    accumulate_results: bool,
    seconds_per_chunk: int | None,
    input_sample_rate: int,
    target_sample_rate: int,
) -> tuple[dict[str, list[float]] | None, float]:
    """Internal function to handle raw PCM mode."""
    print(f"Reading raw PCM from stdin (source sample rate: {input_sample_rate}Hz)...", file=sys.stderr)

    if input_sample_rate != target_sample_rate:
        print(f"Resampling from {input_sample_rate}Hz to {target_sample_rate}Hz...", file=sys.stderr)

    # Create stream wrapper that handles resampling
    stream_wrapper = _RawPcmStreamWrapper(input_sample_rate, target_sample_rate)

    audio_name = "stdin"
    print(f"Finding pattern in audio file {audio_name}...", file=sys.stderr)

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
        )
        .find_clip_in_audio(
            full_streaming_audio,
            on_pattern_detected=on_pattern_detected,
            accumulate_results=accumulate_results,
        )
    )

    return peak_times, total_time


def _make_jsonl_callback() -> PatternDetectedCallback:
    """Create a callback that emits pattern_detected JSONL events."""
    def callback(clip_name: str, timestamp: float) -> None:
        _emit_jsonl(
            "pattern_detected",
            clip_name=clip_name,
            timestamp=timestamp,
            timestamp_formatted=seconds_to_time(timestamp),
        )
    return callback


def _run_match_with_output(
    args: argparse.Namespace,
    pattern_files: list[str],
    audio_source: str | None,
    debug_output_file: str,
    from_stdin: bool = False,
    raw_pcm: bool = False,
    seconds_per_chunk: int | None = 60,
    source_sample_rate: int | None = None,
    target_sample_rate: int | None = None,
) -> tuple[dict[str, list[float]] | None, float]:
    """Run match_pattern and handle output (JSON or JSONL).

    For stdin mode, always uses JSONL output.
    For file mode, uses JSON unless --jsonl is specified.
    """
    # stdin mode always uses JSONL
    jsonl_mode = from_stdin or getattr(args, 'jsonl', False)

    # Create callback for JSONL mode
    callback = _make_jsonl_callback() if jsonl_mode else None

    # Emit start event for JSONL mode
    if jsonl_mode:
        _emit_jsonl("start", source="stdin" if from_stdin else (audio_source or "unknown"))

    # In JSONL mode, don't accumulate results (saves memory)
    peak_times, total_time = match_pattern(
        audio_source,
        pattern_files,
        debug_mode=args.debug,
        on_pattern_detected=callback,
        accumulate_results=not jsonl_mode,
        seconds_per_chunk=seconds_per_chunk,
        from_stdin=from_stdin,
        raw_pcm=raw_pcm,
        sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
    )
    print(f"Total time processed: {seconds_to_time(seconds=total_time)}", file=sys.stderr)

    # In debug mode, also write to file (only if we accumulated results)
    if args.debug and peak_times is not None:
        os.makedirs('./tmp', exist_ok=True)
        with open(debug_output_file, 'w') as f:
            print(json.dumps(peak_times, ensure_ascii=False), file=f)
        print(f"Debug output written to {debug_output_file}", file=sys.stderr)

    # Output
    if jsonl_mode:
        _emit_jsonl("end", total_time=total_time, total_time_formatted=seconds_to_time(total_time))
    else:
        print(json.dumps(peak_times, ensure_ascii=False))

    return peak_times, total_time


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

    # Handle multiplexed stdin mode (patterns + audio all from stdin)
    multiplexed_stdin = getattr(args, 'multiplexed_stdin', False)
    if multiplexed_stdin:
        raw_pcm = getattr(args, 'raw_pcm', False)
        source_sample_rate = getattr(args, 'source_sample_rate', None)

        # Validate raw_pcm mode requires source_sample_rate
        if raw_pcm and source_sample_rate is None:
            print("Error: --source-sample-rate is required when using --raw-pcm", file=sys.stderr)
            sys.exit(1)

        # source_sample_rate should not be used with WAV mode
        if not raw_pcm and source_sample_rate is not None:
            print("Error: --source-sample-rate can only be used with --raw-pcm (WAV mode reads sample rate from header)", file=sys.stderr)
            sys.exit(1)

        # Multiplexed stdin mode: always JSONL output
        callback = _make_jsonl_callback()
        _emit_jsonl("start", source="multiplexed-stdin")

        peak_times, total_time = _match_pattern_multiplexed_stdin(
            debug_mode=args.debug,
            on_pattern_detected=callback,
            accumulate_results=False,
            seconds_per_chunk=seconds_per_chunk,
            raw_pcm=raw_pcm,
            input_sample_rate=source_sample_rate,
            target_sample_rate=sr,
        )

        print(f"Total time processed: {seconds_to_time(seconds=total_time)}", file=sys.stderr)
        _emit_jsonl("end", total_time=total_time, total_time_formatted=seconds_to_time(total_time))
        return

    # Non-multiplexed modes: require pattern file(s)
    if args.pattern_folder:
        pattern_files = []
        for pattern_file in glob.glob(f'{args.pattern_folder}/*.wav'):
            print(f"adding pattern file {pattern_file}...", file=sys.stderr)
            pattern_files.append(pattern_file)
    elif args.pattern_file:
        pattern_files = [args.pattern_file]
    else:
        print("Please provide either --pattern-file, --pattern-folder, or --multiplexed-stdin", file=sys.stderr)
        sys.exit(1)

    jsonl_mode = getattr(args, 'jsonl', False)

    if args.audio_folder:
        if jsonl_mode:
            print("Error: --jsonl is not supported with --audio-folder", file=sys.stderr)
            sys.exit(1)

        print(f"Finding pattern in audio files in folder {args.audio_folder}...", file=sys.stderr)
        all_results = {}
        for audio_file in glob.glob(f'{args.audio_folder}/*.m4a'):
            print(f"Processing {audio_file}...", file=sys.stderr)
            peak_times, total_time = match_pattern(audio_file, pattern_files, debug_mode=args.debug, seconds_per_chunk=seconds_per_chunk, target_sample_rate=target_sample_rate)
            print(f"Total time processed: {seconds_to_time(seconds=total_time)}", file=sys.stderr)
            all_results[audio_file] = peak_times

        # In debug mode, also write to file
        if args.debug:
            output_file = f'./tmp/{os.path.basename(args.audio_folder)}.json'
            os.makedirs('./tmp', exist_ok=True)
            with open(output_file, 'w') as f:
                print(json.dumps(all_results, ensure_ascii=False), file=f)
            print(f"Debug output written to {output_file}", file=sys.stderr)

        # Output final JSON to stdout for piping
        print(json.dumps(all_results, ensure_ascii=False))
    elif args.audio_file:
        _run_match_with_output(
            args, pattern_files, args.audio_file,
            debug_output_file=f'./tmp/{Path(args.audio_file).stem}.json',
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=target_sample_rate,
        )
    elif args.stdin:
        # Stdin mode: always outputs JSONL
        raw_pcm = getattr(args, 'raw_pcm', False)
        source_sample_rate = getattr(args, 'source_sample_rate', None)

        # Validate raw_pcm mode requires source_sample_rate
        if raw_pcm and source_sample_rate is None:
            print("Error: --source-sample-rate is required when using --raw-pcm", file=sys.stderr)
            sys.exit(1)

        # source_sample_rate should not be used with WAV mode
        if not raw_pcm and source_sample_rate is not None:
            print("Error: --source-sample-rate can only be used with --raw-pcm (WAV mode reads sample rate from header)", file=sys.stderr)
            sys.exit(1)

        _run_match_with_output(
            args, pattern_files, None,
            debug_output_file='./tmp/stdin_stream.json',
            from_stdin=True,
            raw_pcm=raw_pcm,
            source_sample_rate=source_sample_rate,
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=target_sample_rate,
        )
    else:
        print("Please provide --audio-file, --audio-folder, --stdin, or --multiplexed-stdin", file=sys.stderr)
        sys.exit(1)


def cmd_show_config(args: argparse.Namespace) -> None:
    """Handler for show-config subcommand"""
    # Get target sample rate (None means use default 8000)
    target_sample_rate = getattr(args, 'target_sample_rate', None)

    if args.pattern_folder:
        pattern_files = []
        for pattern_file in glob.glob(f'{args.pattern_folder}/*.wav'):
            pattern_files.append(pattern_file)
    elif args.pattern_file:
        pattern_files = [args.pattern_file]
    else:
        print("Please provide either --pattern-file or --pattern-folder", file=sys.stderr)
        sys.exit(1)

    pattern_clips = []
    for pattern_file in pattern_files:
        if not os.path.exists(pattern_file):
            print(f"Error: Pattern {pattern_file} does not exist", file=sys.stderr)
            sys.exit(1)
        pattern_clips.append(AudioClip.from_audio_file(pattern_file, sample_rate=target_sample_rate))

    # Use auto mode (None) to show minimum computed values
    detector = AudioPatternDetector(
        audio_clips=pattern_clips,
        debug_mode=False,
        seconds_per_chunk=None,
        target_sample_rate=target_sample_rate,
    )
    config = detector.get_config()
    print(json.dumps(config, indent=2, ensure_ascii=False))
