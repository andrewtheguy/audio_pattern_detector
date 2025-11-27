import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import (
    ffmpeg_get_float32_pcm,
    resample_audio,
    seconds_to_time,
    DEFAULT_TARGET_SAMPLE_RATE,
)

def _emit_jsonl(event_type: str, **kwargs):
    """Emit a JSONL event to stdout and flush immediately."""
    event = {"type": event_type, **kwargs}
    print(json.dumps(event, ensure_ascii=False), flush=True)


def match_pattern(
    audio_source,
    pattern_files: list[str],
    debug_mode=False,
    on_pattern_detected=None,
    accumulate_results=True,
    seconds_per_chunk=60,
    from_stdin=False,
    sample_rate=None,
    target_sample_rate=None,
):
    """Find pattern matches in audio file or stdin (raw PCM)

    Args:
        audio_source: Path to audio file, or None if from_stdin=True
        pattern_files: List of pattern file paths
        debug_mode: Enable debug mode
        on_pattern_detected: Optional callback for streaming output.
                             Signature: on_pattern_detected(clip_name: str, timestamp: float)
        accumulate_results: If False, don't accumulate results (saves memory for streaming)
        seconds_per_chunk: Seconds per chunk for sliding window (None for auto-compute)
        from_stdin: Whether to read raw float32 PCM from stdin
        sample_rate: Sample rate of stdin input (default: target_sample_rate)
        target_sample_rate: Target sample rate for processing (default: DEFAULT_TARGET_SAMPLE_RATE, 8000)
    """
    if not from_stdin and not os.path.exists(audio_source):
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
        # Stdin mode: read raw float32 little-endian PCM directly
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

    # File mode: use ffmpeg
    with ffmpeg_get_float32_pcm(
        audio_source,
        target_sample_rate=sr,
        ac=1,
    ) as stdout:
        audio_name = Path(audio_source).stem
        print(f"Finding pattern in audio file {audio_name}...", file=sys.stderr)
        full_streaming_audio = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)
        # Find clip occurrences in the full audio
        peak_times, total_time = (AudioPatternDetector(debug_mode=debug_mode, audio_clips=pattern_clips, seconds_per_chunk=seconds_per_chunk, target_sample_rate=sr)
                      .find_clip_in_audio(
                          full_streaming_audio,
                          on_pattern_detected=on_pattern_detected,
                          accumulate_results=accumulate_results,
                      ))
    return peak_times, total_time


class _RawPcmStreamWrapper:
    """Wrapper to make raw PCM stdin work like AudioStream expects.

    Reads raw float32 PCM from stdin, optionally resampling to target rate.
    """

    def __init__(self, input_sample_rate: int, target_sample_rate: int):
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.needs_resample = input_sample_rate != target_sample_rate
        # Calculate chunk size for reading (read in chunks that result in target chunk size after resampling)
        self._bytes_per_sample = 4  # float32

    def read(self, size: int) -> bytes:
        """Read and optionally resample audio data.

        Args:
            size: Number of bytes to return (at target sample rate)

        Returns:
            Raw bytes of float32 PCM at target sample rate
        """
        if not self.needs_resample:
            return sys.stdin.buffer.read(size)

        # Calculate how many input bytes we need to produce the requested output bytes
        target_samples = size // self._bytes_per_sample
        input_samples = int(target_samples * self.input_sample_rate / self.target_sample_rate)
        input_bytes = input_samples * self._bytes_per_sample

        # Read input data
        data = sys.stdin.buffer.read(input_bytes)
        if not data:
            return b""

        # Convert to numpy, resample, and convert back to bytes
        audio = np.frombuffer(data, dtype=np.float32)
        resampled = resample_audio(audio, self.input_sample_rate, self.target_sample_rate)
        return resampled.tobytes()


def _match_pattern_raw_pcm(
    pattern_clips,
    debug_mode,
    on_pattern_detected,
    accumulate_results,
    seconds_per_chunk,
    input_sample_rate,
    target_sample_rate,
):
    """Internal function to handle raw PCM mode."""
    print(f"Reading raw PCM from stdin (sample rate: {input_sample_rate}Hz)...", file=sys.stderr)

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


def _make_jsonl_callback():
    """Create a callback that emits pattern_detected JSONL events."""
    def callback(clip_name: str, timestamp: float):
        _emit_jsonl(
            "pattern_detected",
            clip_name=clip_name,
            timestamp=timestamp,
            timestamp_formatted=seconds_to_time(timestamp),
        )
    return callback


def _run_match_with_output(
    args, pattern_files, audio_source, debug_output_file,
    from_stdin=False, seconds_per_chunk=60, sample_rate=None, target_sample_rate=None
):
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
        sample_rate=sample_rate,
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


def cmd_match(args):
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

    if args.pattern_folder:
        pattern_files = []
        for pattern_file in glob.glob(f'{args.pattern_folder}/*.wav'):
            print(f"adding pattern file {pattern_file}...", file=sys.stderr)
            pattern_files.append(pattern_file)
    elif args.pattern_file:
        pattern_files = [args.pattern_file]
    else:
        print("Please provide either --pattern-file or --pattern-folder", file=sys.stderr)
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
        # Stdin mode: raw float32 PCM, always outputs JSONL
        sample_rate = getattr(args, 'sample_rate', None)
        _run_match_with_output(
            args, pattern_files, None,
            debug_output_file='./tmp/stdin_stream.json',
            from_stdin=True,
            sample_rate=sample_rate,
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=target_sample_rate,
        )
    else:
        print("Please provide --audio-file, --audio-folder, or --stdin", file=sys.stderr)
        sys.exit(1)


def cmd_show_config(args):
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
