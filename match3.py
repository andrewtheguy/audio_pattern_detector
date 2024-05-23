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
from audio_offset_finder_v2 import find_clip_in_audio_in_chunks, DEFAULT_METHOD, cleanup_peak_times, set_debug_mode
from andrew_utils import seconds_to_time

def main():
    set_debug_mode(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-file', metavar='pattern file', required=True, type=str, help='pattern file')
    parser.add_argument('--audio-file', metavar='audio file', type=str, required=True, help='audio file to find pattern')
    parser.add_argument('--match-method', metavar='pattern match method', type=str, help='pattern match method',default=DEFAULT_METHOD)
    parser.add_argument('--threshold', metavar='pattern match method', type=float, help='pattern match method',
                        default=0.4)
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()
    #print(args.method)

    # Find clip occurrences in the full audio
    peak_times = find_clip_in_audio_in_chunks(clip_paths=[args.pattern_file], full_audio_path=args.audio_file, method=args.match_method,
                                              threshold=args.threshold)
    print(peak_times)
    peak_times_clean = cleanup_peak_times(peak_times[args.pattern_file])
    print(peak_times_clean)

    for offset in peak_times_clean:
        print(f"Clip occurs at the following times (in seconds): {seconds_to_time(seconds=offset)}" )
    #    #print(f"Offset: {offset}s" )
    
    distances=[]
    for i in range(1,len(peak_times_clean)):
        hour_delta = peak_times_clean[i]//3600-peak_times_clean[i-1]//3600
        distance = peak_times_clean[i]-peak_times_clean[i-1]
        distances.append({"distance":seconds_to_time(seconds=distance),"hour_delta":hour_delta,"pair":[seconds_to_time(seconds=peak_times_clean[i-1]),seconds_to_time(seconds=peak_times_clean[i])]})
    print("Distances between clips:")    
    pprint.pprint(distances)    

if __name__ == '__main__':
    main()
