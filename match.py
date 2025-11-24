import argparse
import glob
import json
import os
import sys
from pathlib import Path

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_pattern_detector import AudioPatternDetector
from andrew_utils import seconds_to_time

from audio_pattern_detector.audio_utils import ffmpeg_get_16bit_pcm, TARGET_SAMPLE_RATE


# # only for testing
# def cleanup_peak_times(peak_times):
#     # freq = {}

#     # for peak in peak_times:
#     #     i = math.floor(peak)
#     #     cur = freq.get(i, 0)
#     #     freq[i] = cur + 1

#     #print(freq)

#     #print({k: v for k, v in sorted(freq.items(), key=lambda item: item[1])})

#     #print('before consolidate',peak_times)

#     # deduplicate by seconds
#     peak_times_clean = list(dict.fromkeys([math.floor(peak) for peak in peak_times]))

#     peak_times_clean2 = deque(sorted(peak_times_clean))
#     #print('before remove close',peak_times_clean2)

#     peak_times_final = []

#     # skip those less than 10 seconds in between like beep, beep, beep
#     # already doing that in process timestamp, just doing again to
#     # make it look clean
#     skip_second_between = 10

#     prevItem = None
#     while peak_times_clean2:
#         item = peak_times_clean2.popleft()
#         if (prevItem is None):
#             peak_times_final.append(item)
#             prevItem = item
#         elif item - prevItem < skip_second_between:
#             #logger.debug(f'skip {item} less than {skip_second_between} seconds from {prevItem}')
#             prevItem = item
#         else:
#             peak_times_final.append(item)
#             prevItem = item

#     return peak_times_final

def match_pattern(audio_file, pattern_files: list[str], debug_mode=False):
    if not os.path.exists(audio_file):
        raise ValueError(f"Audio {audio_file} does not exist")

    pattern_clips = []
    for pattern_file in pattern_files:
        if not os.path.exists(pattern_file):
            raise ValueError(f"Pattern {pattern_file} does not exist")
        pattern_clip = AudioClip.from_audio_file(pattern_file)
        pattern_clips.append(pattern_clip)

    if len(pattern_clips)  == 0:
        raise ValueError("No pattern clips passed")

    sr = TARGET_SAMPLE_RATE
    with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=sr, ac=1) as stdout:
        audio_name = Path(audio_file).stem
        print(f"Finding pattern in audio file {audio_name}...",file=sys.stderr)
        #exit(1)
        full_streaming_audio = AudioStream(name=audio_name, audio_stream=stdout, sample_rate=sr)
        # Find clip occurrences in the full audio
        peak_times, total_time = (AudioPatternDetector(debug_mode=debug_mode,audio_clips=pattern_clips)
                      .find_clip_in_audio(full_streaming_audio))
    return peak_times, total_time


def main():
    #set_debug_mode(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-file', metavar='pattern file', required=False, type=str, help='pattern file')
    parser.add_argument('--pattern-folder', metavar='pattern folder', required=False, type=str, help='folder with pattern audio clips')
    parser.add_argument('--audio-file', metavar='audio file', type=str, required=False, help='audio file to find pattern')
    parser.add_argument('--audio-folder', metavar='audio folder', type=str, required=False, help='audio folder to find pattern in files')
    parser.add_argument('--debug', metavar='debug', action=argparse.BooleanOptionalAction, help='debug mode (audio file only)', default=True)
    #parser.add_argument('--threshold', metavar='pattern match method', type=float, help='pattern match method',
    #                    default=0.4)
    args = parser.parse_args()

    if args.pattern_folder:
        pattern_files = []
        for pattern_file in glob.glob(f'{args.pattern_folder}/*.wav'):
            print(f"adding pattern file {pattern_file}...",file=sys.stderr)
            pattern_files.append(pattern_file)
    elif args.pattern_file:
        pattern_files = [args.pattern_file]
    else:
        print("Please provide either --pattern-file or --pattern-folder",file=sys.stderr)
        exit(1)

    if args.audio_folder:
        output_file_prefix=f'{os.path.basename(args.audio_folder)}'
        output_file=f'./tmp/{output_file_prefix}.jsonl'
        with open(output_file, 'w') as f:
            f.truncate(0)
        print(f"Finding pattern in audio files in folder {args.audio_folder}...",file=sys.stderr)
        #peak_time = {}
        for audio_file in glob.glob(f'{args.audio_folder}/*.m4a'):
            print(f"Processing {audio_file}...",file=sys.stderr)
            peak_times, total_time = match_pattern(audio_file, pattern_files, debug_mode=False)
            print(peak_times,file=sys.stderr)
            print(f"Total time processed: {seconds_to_time(seconds=total_time)}",file=sys.stderr)
            if len(peak_times) > 0:
                peak_times_second = [seconds_to_time(seconds=offset) for offset in peak_times]
                print(f"Clip occurs with the file {audio_file} at the following times (in seconds): {peak_times_second}",file=sys.stderr)
                #all_files[audio_file] = peak_times_second
                with open(output_file, 'a') as f:
                    print(json.dumps({'audio_file': audio_file, 'peak_times': peak_times_second},ensure_ascii=False), file=f)
    elif args.audio_file:
        peak_times,total_time=match_pattern(args.audio_file, pattern_files, debug_mode=args.debug)
        print(peak_times,file=sys.stderr)
        print(f"Total time processed: {seconds_to_time(seconds=total_time)}",file=sys.stderr)
        output_file = f'./tmp/{Path(args.audio_file).stem}.json'
        with open(output_file, 'w') as f:
            print(json.dumps({'audio_file': args.audio_file, 'peak_times': peak_times},ensure_ascii=False), file=f)
    else:
        print("Please provide either --audio-file or --audio-folder",file=sys.stderr)
        exit(1)


if __name__ == '__main__':
    main()
