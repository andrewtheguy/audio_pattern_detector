import glob
import json
import os
import sys
from pathlib import Path

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from audio_pattern_detector.audio_utils import (
    ffmpeg_get_16bit_pcm,
    TARGET_SAMPLE_RATE
)
from audio_pattern_detector.audio_utils import seconds_to_time

def match_pattern(audio_file, pattern_files: list[str], debug_mode=False):
    """Find pattern matches in audio file"""
    if not os.path.exists(audio_file):
        raise ValueError(f"Audio {audio_file} does not exist")

    pattern_clips = []
    for pattern_file in pattern_files:
        if not os.path.exists(pattern_file):
            raise ValueError(f"Pattern {pattern_file} does not exist")
        pattern_clip = AudioClip.from_audio_file(pattern_file)
        pattern_clips.append(pattern_clip)

    if len(pattern_clips) == 0:
        raise ValueError("No pattern clips passed")

    sr = TARGET_SAMPLE_RATE
    with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
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
        output_file_prefix = f'{os.path.basename(args.audio_folder)}'
        output_file = f'./tmp/{output_file_prefix}.jsonl'
        with open(output_file, 'w') as f:
            f.truncate(0)
        print(f"Finding pattern in audio files in folder {args.audio_folder}...", file=sys.stderr)
        for audio_file in glob.glob(f'{args.audio_folder}/*.m4a'):
            print(f"Processing {audio_file}...", file=sys.stderr)
            peak_times, total_time = match_pattern(audio_file, pattern_files, debug_mode=False)
            print(peak_times, file=sys.stderr)
            print(f"Total time processed: {seconds_to_time(seconds=total_time)}", file=sys.stderr)
            if len(peak_times) > 0:
                peak_times_second = [seconds_to_time(seconds=offset) for offset in peak_times]
                print(f"Clip occurs with the file {audio_file} at the following times (in seconds): {peak_times_second}", file=sys.stderr)
                with open(output_file, 'a') as f:
                    print(json.dumps({'audio_file': audio_file, 'peak_times': peak_times_second}, ensure_ascii=False), file=f)
    elif args.audio_file:
        peak_times, total_time = match_pattern(args.audio_file, pattern_files, debug_mode=args.debug)
        print(peak_times, file=sys.stderr)
        print(f"Total time processed: {seconds_to_time(seconds=total_time)}", file=sys.stderr)
        output_file = f'./tmp/{Path(args.audio_file).stem}.json'
        with open(output_file, 'w') as f:
            print(json.dumps({'audio_file': args.audio_file, 'peak_times': peak_times}, ensure_ascii=False), file=f)
    else:
        print("Please provide either --audio-file or --audio-folder", file=sys.stderr)
        sys.exit(1)
