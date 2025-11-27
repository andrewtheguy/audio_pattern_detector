import glob
import json
import os
import sys
from pathlib import Path

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import (
    ffmpeg_get_16bit_pcm,
    get_audio_duration,
    seconds_to_time,
    TARGET_SAMPLE_RATE,
)

def match_pattern(audio_source, pattern_files: list[str], debug_mode=False, is_url=False):
    """Find pattern matches in audio file or URL"""
    if not is_url and not os.path.exists(audio_source):
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
    with ffmpeg_get_16bit_pcm(audio_source, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_source).stem if not is_url else "stream"
        print(f"Finding pattern in audio file {audio_name}...", file=sys.stderr)
        full_streaming_audio = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)
        # Find clip occurrences in the full audio
        peak_times, total_time = (AudioPatternDetector(debug_mode=debug_mode, audio_clips=pattern_clips)
                      .find_clip_in_audio(full_streaming_audio))
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

    if args.audio_folder:
        print(f"Finding pattern in audio files in folder {args.audio_folder}...", file=sys.stderr)
        all_results = {}
        for audio_file in glob.glob(f'{args.audio_folder}/*.m4a'):
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
        peak_times, total_time = match_pattern(args.audio_file, pattern_files, debug_mode=args.debug)
        print(f"Total time processed: {seconds_to_time(seconds=total_time)}", file=sys.stderr)

        # In debug mode, also write to file
        if args.debug:
            output_file = f'./tmp/{Path(args.audio_file).stem}.json'
            os.makedirs('./tmp', exist_ok=True)
            with open(output_file, 'w') as f:
                print(json.dumps(peak_times, ensure_ascii=False), file=f)
            print(f"Debug output written to {output_file}", file=sys.stderr)

        # Output final JSON to stdout for piping
        print(json.dumps(peak_times, ensure_ascii=False))
    elif args.audio_url:
        # Validate URL has a duration (not a live stream)
        print(f"Checking URL duration: {args.audio_url}...", file=sys.stderr)
        duration = get_audio_duration(args.audio_url)
        if duration is None:
            print("Error: URL appears to be a live stream (no duration). Only non-live audio is supported.", file=sys.stderr)
            sys.exit(1)
        print(f"URL duration: {seconds_to_time(seconds=duration)}", file=sys.stderr)

        peak_times, total_time = match_pattern(args.audio_url, pattern_files, debug_mode=args.debug, is_url=True)
        print(f"Total time processed: {seconds_to_time(seconds=total_time)}", file=sys.stderr)

        # In debug mode, also write to file
        if args.debug:
            output_file = './tmp/url_stream.json'
            os.makedirs('./tmp', exist_ok=True)
            with open(output_file, 'w') as f:
                print(json.dumps(peak_times, ensure_ascii=False), file=f)
            print(f"Debug output written to {output_file}", file=sys.stderr)

        # Output final JSON to stdout for piping
        print(json.dumps(peak_times, ensure_ascii=False))
    else:
        print("Please provide --audio-file, --audio-folder, or --audio-url", file=sys.stderr)
        sys.exit(1)
