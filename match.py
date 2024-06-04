import argparse
from collections import deque
import copy
import datetime
import pdb
import pprint
import time
import librosa
import numpy as np
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

import ffmpeg
import librosa
import soundfile as sf
from audio_offset_finder_v2 import DEFAULT_METHOD, AudioOffsetFinder
from andrew_utils import seconds_to_time


# only for testing
def cleanup_peak_times(peak_times):
    # freq = {}

    # for peak in peak_times:
    #     i = math.floor(peak)
    #     cur = freq.get(i, 0)
    #     freq[i] = cur + 1

    #print(freq)

    #print({k: v for k, v in sorted(freq.items(), key=lambda item: item[1])})

    #print('before consolidate',peak_times)

    # deduplicate by seconds
    peak_times_clean = list(dict.fromkeys([math.floor(peak) for peak in peak_times]))

    peak_times_clean2 = deque(sorted(peak_times_clean))
    #print('before remove close',peak_times_clean2)

    peak_times_final = []

    # skip those less than 10 seconds in between like beep, beep, beep
    # already doing that in process timestamp, just doing again to
    # make it look clean
    skip_second_between = 10

    prevItem = None
    while peak_times_clean2:
        item = peak_times_clean2.popleft()
        if (prevItem is None):
            peak_times_final.append(item)
            prevItem = item
        elif item - prevItem < skip_second_between:
            #logger.debug(f'skip {item} less than {skip_second_between} seconds from {prevItem}')
            prevItem = item
        else:
            peak_times_final.append(item)
            prevItem = item

    return peak_times_final

def match_pattern(audio_file, pattern_file, method):
    # Find clip occurrences in the full audio
    peak_times = AudioOffsetFinder(method=method, debug_mode=True,
                                   clip_paths=[pattern_file]).find_clip_in_audio(full_audio_path=audio_file)
    return peak_times[pattern_file]


def main():
    #set_debug_mode(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-file', metavar='pattern file', required=True, type=str, help='pattern file')
    parser.add_argument('--audio-file', metavar='audio file', type=str, required=True, help='audio file to find pattern')
    parser.add_argument('--match-method', metavar='pattern match method', type=str, help='pattern match method, currently only correlation',default=DEFAULT_METHOD)
    #parser.add_argument('--threshold', metavar='pattern match method', type=float, help='pattern match method',
    #                    default=0.4)
    args = parser.parse_args()
    peak_time=match_pattern(args.audio_file, args.pattern_file, args.match_method)
    print(peak_time)

    for offset in peak_time:
        print(f"Clip occurs at the following times (in seconds): {seconds_to_time(seconds=offset)}")


if __name__ == '__main__':
    main()
