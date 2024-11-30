import argparse
import glob
import json
import os
from pathlib import Path

from audio_offset_finder_v2.audio_offset_finder_v2 import AudioOffsetFinder
from andrew_utils import seconds_to_time


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

def match_pattern(audio_file, pattern_file, debug_mode=False):
    # Find clip occurrences in the full audio
    peak_times = AudioOffsetFinder(debug_mode=debug_mode,
                                   clip_paths=[pattern_file]).find_clip_in_audio(full_audio_path=audio_file)
    return peak_times[pattern_file]


def main():
    #set_debug_mode(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-file', metavar='pattern file', required=True, type=str, help='pattern file')
    parser.add_argument('--audio-file', metavar='audio file', type=str, required=False, help='audio file to find pattern')
    parser.add_argument('--audio-folder', metavar='audio folder', type=str, required=False, help='audio folder to find pattern in files')
    #parser.add_argument('--threshold', metavar='pattern match method', type=float, help='pattern match method',
    #                    default=0.4)
    args = parser.parse_args()

    if args.audio_folder:
        #basename(args.pattern_file)
        output_file_prefix=Path(args.pattern_file).stem
        output_file_prefix=f'{os.path.basename(args.audio_folder)}_{output_file_prefix}'
        output_file=f'./tmp/{output_file_prefix}.jsonl'
        with open(output_file, 'w') as f:
            f.truncate(0)
        print(f"Finding pattern in audio files in folder {args.audio_folder}...")
        #peak_time = {}
        for audio_file in glob.glob(f'{args.audio_folder}/*.m4a'):
            print(f"Processing {audio_file}...")
            peak_times = match_pattern(audio_file, args.pattern_file)
            if len(peak_times) > 0:
                peak_times_second = [seconds_to_time(seconds=offset) for offset in peak_times]
                print(f"Clip occurs with the file {audio_file} at the following times (in seconds): {peak_times_second}")
                #all_files[audio_file] = peak_times_second
                with open(output_file, 'a') as f:
                    print(json.dumps({'audio_file': audio_file, 'peak_times': peak_times_second},ensure_ascii=False), file=f)
    elif args.audio_file:
        peak_times=match_pattern(args.audio_file, args.pattern_file, debug_mode=True)
        print(peak_times)

        for offset in peak_times:
            print(f"Clip occurs at the following times (in seconds): {seconds_to_time(seconds=offset)}")
    else:
        print("Please provide either --audio-file or --audio-folder")
        exit(1)


if __name__ == '__main__':
    main()
