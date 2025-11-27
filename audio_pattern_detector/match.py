import glob
import json
import os
import sys
from pathlib import Path

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import (
    ffmpeg_get_float32_pcm,
    get_audio_duration,
    seconds_to_time,
    TARGET_SAMPLE_RATE,
)

def _emit_jsonl(event_type: str, **kwargs):
    """Emit a JSONL event to stdout and flush immediately."""
    event = {"type": event_type, **kwargs}
    print(json.dumps(event, ensure_ascii=False), flush=True)


def match_pattern(
    audio_source,
    pattern_files: list[str],
    debug_mode=False,
    is_url=False,
    from_stdin=False,
    input_format=None,
    on_pattern_detected=None,
    accumulate_results=True,
):
    """Find pattern matches in audio file, URL, or stdin

    Args:
        audio_source: Path to audio file, URL, or None if from_stdin=True
        pattern_files: List of pattern file paths
        debug_mode: Enable debug mode
        is_url: Whether audio_source is a URL
        from_stdin: Whether to read audio from stdin
        input_format: Input format hint for ffmpeg when reading from stdin
        on_pattern_detected: Optional callback for streaming output.
                             Signature: on_pattern_detected(clip_name: str, timestamp: float)
        accumulate_results: If False, don't accumulate results (saves memory for streaming)
    """
    if not is_url and not from_stdin and not os.path.exists(audio_source):
        raise ValueError(f"Audio {audio_source} does not exist")

    pattern_clips = []
    for pattern_file in pattern_files:
        if not os.path.exists(pattern_file):
            raise ValueError(f"Pattern {pattern_file} does not exist")
        pattern_clip = AudioClip.from_audio_file(pattern_file)
        pattern_clips.append(pattern_clip)

    if len(pattern_clips) == 0:
        raise ValueError("No pattern clips passed")

    sr = TARGET_SAMPLE_RATE
    with ffmpeg_get_float32_pcm(
        audio_source,
        target_sample_rate=sr,
        ac=1,
        from_stdin=from_stdin,
        input_format=input_format,
    ) as stdout:
        if from_stdin:
            audio_name = "stdin"
        elif is_url:
            audio_name = "stream"
        else:
            audio_name = Path(audio_source).stem
        print(f"Finding pattern in audio file {audio_name}...", file=sys.stderr)
        full_streaming_audio = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)
        # Find clip occurrences in the full audio
        peak_times, total_time = (AudioPatternDetector(debug_mode=debug_mode, audio_clips=pattern_clips)
                      .find_clip_in_audio(
                          full_streaming_audio,
                          on_pattern_detected=on_pattern_detected,
                          accumulate_results=accumulate_results,
                      ))
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
    is_url=False, from_stdin=False, input_format=None
):
    """Run match_pattern and handle output (JSON or JSONL)."""
    jsonl_mode = getattr(args, 'jsonl', False)

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
        is_url=is_url,
        from_stdin=from_stdin,
        input_format=input_format,
        on_pattern_detected=callback,
        accumulate_results=not jsonl_mode,
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
        # Support multiple audio formats
        audio_files = []
        for ext in ['*.m4a', '*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(glob.glob(f'{args.audio_folder}/{ext}'))
        # Sort for consistent ordering
        audio_files.sort()
        for audio_file in audio_files:
            print(f"Processing {audio_file}...", file=sys.stderr)
            peak_times, total_time = match_pattern(audio_file, pattern_files, debug_mode=args.debug)
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
        )
    elif args.audio_url:
        # Validate URL has a duration (not a live stream)
        print(f"Checking URL duration: {args.audio_url}...", file=sys.stderr)
        duration = get_audio_duration(args.audio_url)
        if duration is None:
            print("Error: URL appears to be a live stream (no duration). Only non-live audio is supported.", file=sys.stderr)
            sys.exit(1)
        print(f"URL duration: {seconds_to_time(seconds=duration)}", file=sys.stderr)

        _run_match_with_output(
            args, pattern_files, args.audio_url,
            debug_output_file='./tmp/url_stream.json',
            is_url=True,
        )
    elif args.stdin:
        input_format = getattr(args, 'input_format', None)
        _run_match_with_output(
            args, pattern_files, None,
            debug_output_file='./tmp/stdin_stream.json',
            from_stdin=True,
            input_format=input_format,
        )
    else:
        print("Please provide --audio-file, --audio-folder, --audio-url, or --stdin", file=sys.stderr)
        sys.exit(1)
