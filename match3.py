import argparse
from collections import deque
import copy
import datetime
import pdb
import time
import librosa
import numpy as np
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

import ffmpeg
import librosa
import soundfile as sf
from audio_offset_finder_v2 import find_clip_in_audio_in_chunks
from andrew_utils import seconds_to_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-file', metavar='pattern file', type=str, help='pattern file')
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')
    parser.add_argument('--match-method', metavar='pattern match method', type=str, help='pattern match method',default="correlation")
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()
    #print(args.method)

    # Find clip occurrences in the full audio
    peak_times = find_clip_in_audio_in_chunks(args.pattern_file, args.audio_file, method=args.match_method)
    print(peak_times)

    for offset in peak_times:
        print(f"Clip occurs at the following times (in seconds): {seconds_to_time(seconds=offset,include_decimals=False)}" )
    #    #print(f"Offset: {offset}s" )
    

if __name__ == '__main__':
    main()
