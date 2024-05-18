import argparse
from collections import deque
import copy
import datetime
import json
import logging
import os
import pdb
import time
import librosa
import numpy as np
import scipy
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

import ffmpeg
import librosa
import soundfile as sf
import pyloudnorm as pyln
#import pyaudio

import warnings
from scipy.io import wavfile
from scipy.signal import stft, istft
from andrew_utils import seconds_to_time
from scipy.signal import resample
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)

#ignore possible clipping
warnings.filterwarnings('ignore', module='pyloudnorm')

DEFAULT_METHOD="correlation"

target_sample_rate = 8000

plot_test_x = np.array([])
plot_test_y = np.array([])

debug_mode = False

def load_audio_file(file_path, sr=None):
    # Create ffmpeg process
    process = (
        ffmpeg
        .input(file_path)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=sr, loglevel="error")
        .run_async(pipe_stdout=True)
    )
    data = process.stdout.read()
    process.wait()
    return np.frombuffer(data, dtype="int16")
    #return librosa.load(file_path, sr=sr, mono=True)  # mono=True ensures a single channel audio




def downsample(factor, values):
    buffer_ = deque([], maxlen=factor)
    downsampled_values = []
    for i, value in enumerate(values):
        buffer_.appendleft(value)
        if (i - 1) % factor == 0:
            # Take max value out of buffer
            # or you can take higher value if their difference is too big, otherwise just average
            downsampled_values.append(max(buffer_))
    return np.array(downsampled_values)




def compute_mod_z_score(data):
    """
    Calculates the modified z-score for a 1D array of data.

    Args:
      data: A 1D numpy array of numerical data.

    Returns:
      A 1D numpy array containing the modified z-scores for each data point.
    """
    median = np.median(data)
    median_absolute_deviation = np.median(np.abs(data - median))

    if median_absolute_deviation == 0:
        return np.zeros_like(data)  # Avoid division by zero

    modified_z_scores = 0.6745 * (data - median) / median_absolute_deviation
    return modified_z_scores

def max_distance(sorted_data):
    max_dist = 0
    for i in range(1, len(sorted_data)):
        dist = sorted_data[i] - sorted_data[i - 1]
        max_dist = max(max_dist, dist)
    return max_dist


def correlation_method(clip, audio_section, sr, index, seconds_per_chunk, clip_name,threshold):

    clip_length = len(clip)

    zeroes_second_pad = 1
    # pad zeros between audio and clip
    zeroes = np.zeros(clip_length + zeroes_second_pad * sr)
    audio = np.concatenate((audio_section, zeroes, clip))
    samples_skip_end = zeroes_second_pad * sr + clip_length

    # Cross-correlate and normalize correlation
    correlation = correlate(audio, clip, mode='full', method='fft')
    # abs
    correlation = np.abs(correlation)
    # alternative to replace negative values with zero in array instead of above
    #correlation[correlation < 0] = 0
    correlation /= np.max(correlation)

    section_label = seconds_to_time(index*60)
    if debug_mode:
        graph_dir = f"./tmp/graph/cross_correlation/{clip_name}"
        os.makedirs(graph_dir, exist_ok=True)

        #Optional: plot the correlation graph to visualize
        plt.figure(figsize=(10, 4))
        plt.plot(correlation)
        plt.title('Cross-correlation between the audio clip and full track before slicing')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coefficient')
        plt.savefig(
            f'{graph_dir}/{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.png')
        plt.close()

    max_sample = len(audio) - samples_skip_end
    #trim placeholder clip
    correlation = correlation[:max_sample]
    if debug_mode:
        print(f"{section_label} percentile", np.percentile(correlation, 99))
        print(f"{section_label} max correlation: {max(correlation)}")
        print(f"---")

    height = threshold
    distance = clip_length
    # find the peaks in the spectrogram
    peaks, properties = find_peaks(correlation, height=height, distance=distance)

    section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)

    if debug_mode:
        peak_dir = f"./tmp/peaks/cross_correlation_{clip_name}"
        os.makedirs(peak_dir, exist_ok=True)
        peaks_test=[]
        for item in (peaks):
            #plot_test_x=np.append(plot_test_x, index)
            #plot_test_y=np.append(plot_test_y, item)
            peaks_test.append([int(item),item/sr,correlation[item]])
        print(json.dumps(peaks_test, indent=2), file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))

    #peaks = np.where(correlation > threshold)[0]

    peak_times = np.array(peaks) / sr

    #
    # max_time = ((len(audio) - 1) - samples_skip_end) / sr
    #
    # # Array operation to filter peak_times
    # peak_times2 = peak_times[(peak_times >= 0) & (peak_times <= max_time)]

    return peak_times



# sliding_window: for previous_chunk in seconds from end
# index: for debugging by saving a file for audio_section
# seconds_per_chunk: default seconds_per_chunk
def process_chunk(chunk, clip, sr, previous_chunk, sliding_window, index, seconds_per_chunk, clip_name,
                  method,correlation_threshold):
    clip_length = len(clip)
    new_seconds = len(chunk) / sr
    # Concatenate previous chunk for continuity in processing
    if previous_chunk is not None:
        if new_seconds < seconds_per_chunk:  # too small
            # no need for sliding window since it is the last piece
            subtract_seconds = -(new_seconds - seconds_per_chunk)
            audio_section_temp = np.concatenate((previous_chunk, chunk))[(-seconds_per_chunk * sr):]
            audio_section = np.concatenate((audio_section_temp, np.array([])))
        else:
            subtract_seconds = sliding_window
            #print("sliding_window", sliding_window)
            #print("sr", sr)
            audio_section = np.concatenate((previous_chunk[int(-sliding_window * sr):], chunk, np.array([])))
    else:
        subtract_seconds = 0
        audio_section = np.concatenate((chunk, np.array([])))

    #audio_section = reduce_noise_spectral_subtraction(audio_section)

    #audio_section = audio_section / np.max(np.abs(audio_section))
    # peak normalize audio to -1 dB
    #audio_section = pyln.normalize.peak(audio_section, -1.0)

    normalize = True
    # if normalize:
    #     audio_section = audio_section / np.max(np.abs(audio_section))
    #     clip = clip / np.max(np.abs(clip))

    if normalize:
        audio_section_seconds = len(audio_section) / sr
        #normalize loudness
        if audio_section_seconds < 0.5:
            meter = pyln.Meter(sr, block_size=audio_section_seconds)
        else:
            meter = pyln.Meter(sr)  # create BS.1770 meter

        loudness = meter.integrated_loudness(audio_section)

        # loudness normalize audio to -12 dB LUFS
        audio_section = pyln.normalize.loudness(audio_section, loudness, -12.0)

        clip_second = clip_length / sr

        # normalize loudness
        if clip_second < 0.5:
            meter = pyln.Meter(sr, block_size=clip_second)
        else:
            meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(clip)

        # loudness normalize audio to -12 dB LUFS
        clip = pyln.normalize.loudness(clip, loudness, -12.0)


    # if method == "mfcc":
    #     zeroes_second_pad=1
    #     zeroes = np.zeros(clip_length+zeroes_second_pad*sr)
    #     #pad zeros to the very end
    #     audio_section = np.concatenate((audio_section,zeroes))
    #     samples_skip_end = zeroes_second_pad*sr

    os.makedirs("./tmp/audio", exist_ok=True)
    if debug_mode:
        sf.write(
            f"./tmp/audio/section_{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.wav",
            audio_section, sr)

    if method == "correlation":
        # samples_skip_end does not skip results from being included yet
        peak_times = correlation_method(clip, audio_section=audio_section, sr=sr, index=index,
                                        seconds_per_chunk=seconds_per_chunk,
                                        clip_name=clip_name,threshold=correlation_threshold)
    elif method == "advanced_correlation":
        raise ValueError("disabled")
        peak_times = advanced_correlation_method(clip, audio=audio_section, sr=sr, index=index,
                                        seconds_per_chunk=seconds_per_chunk, clip_name=clip_name)
    elif method == "mfcc":
        raise ValueError("not working well yet")
        peak_times = mfcc_method2(clip, audio=audio_section, sr=sr, index=index,
                                  seconds_per_chunk=seconds_per_chunk,
                                 clip_name=clip_name)
    elif method == "chroma_method":
        raise ValueError("disabled")
        peak_times = chroma_method(clip, audio=audio_section, sr=sr, index=index, seconds_per_chunk=seconds_per_chunk,
                                 clip_name=clip_name)
    else:
        raise ValueError("unknown method")

    #print(peak_times)
    #print(subtract_seconds)

    peak_times_final = [peak_time - subtract_seconds for peak_time in peak_times]
    #peak_times_final = [peak_time for peak_time in peak_times_final if peak_time >= 0]

    #for item in correlation:
    #    if item > threshold:
    #        print(item)
    return peak_times_final

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
            logger.debug(f'skip {item} less than {skip_second_between} seconds from {prevItem}')
            prevItem = item
        else:
            peak_times_final.append(item)
            prevItem = item

    return peak_times_final


def convert_audio_arr_to_float(audio):
    #raise "chafa"
    return librosa.util.buf_to_float(audio,n_bytes=2, dtype='float32')


def convert_audio_to_clip_format(audio_path, output_path):
    if not os.path.exists(audio_path):
        raise ValueError(f"Audio {audio_path} does not exist")
    
    # Load the audio clip
    clip = load_audio_file(audio_path, sr=target_sample_rate)

    # convert to float
    clip = convert_audio_arr_to_float(clip)

    sf.write(output_path, clip, target_sample_rate)

def get_chunking_timing_info(clip_name,clip_seconds,seconds_per_chunk):

    sliding_window = 5

    if (sliding_window < clip_seconds + 5):
        # need to extend the sliding window to overlap the clip
        sliding_window = clip_seconds + 5
        print(f"adjusted sliding_window to {sliding_window} for {clip_name}")
    #print(sliding_window)

    # this should not happen anyways because the seconds per chunk is too small
    if (seconds_per_chunk < sliding_window * 2):
        seconds_per_chunk = sliding_window * 10
        raise ValueError(f"seconds_per_chunk {seconds_per_chunk} is too small")
        #print(f"adjusted seconds_per_chunk to {seconds_per_chunk}")
    #print(seconds_per_chunk)

    #exit(1)

    return sliding_window

# could cause issues with small overlap when intro is followed right by news report
def find_clip_in_audio_in_chunks(clip_paths, full_audio_path, method,correlation_threshold):
    for clip_path in clip_paths:
        if not os.path.exists(clip_path):
            raise ValueError(f"Clip {clip_path} does not exist")
    
    if not os.path.exists(full_audio_path):
        raise ValueError(f"Full audio {full_audio_path} does not exist")

    seconds_per_chunk = 60

    # 2 bytes per channel on every sample for 16 bits (int16)
    # times two because it is (int16, mono)
    chunk_size = (seconds_per_chunk * target_sample_rate) * 2  

    # Initialize parameters

    previous_chunk = None  # Buffer to maintain continuity between chunks

    all_peak_times = {clip_path: [] for clip_path in clip_paths}


    # Create ffmpeg process
    process = (
        ffmpeg
        .input(full_audio_path)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=target_sample_rate, loglevel="error")
        .run_async(pipe_stdout=True)
    )
    #audio_size = 0

    i = 0

    clip_datas={}

    for clip_path in clip_paths:
        # Load the audio clip
        clip = load_audio_file(clip_path, sr=target_sample_rate)
        # convert to float
        clip = convert_audio_arr_to_float(clip)

        clip_name, _ = os.path.splitext(os.path.basename(clip_path))

        clip_seconds = len(clip) / target_sample_rate
        
        sliding_window = get_chunking_timing_info(clip_name,clip_seconds,seconds_per_chunk)

        clip_datas[clip_path] = {"clip":clip,"clip_name":clip_name,"clip_seconds":clip_seconds,"sliding_window":sliding_window}

    # Process audio in chunks
    while True:
        in_bytes = process.stdout.read(chunk_size)
        if not in_bytes:
            break
        # Convert bytes to numpy array
        chunk = np.frombuffer(in_bytes, dtype="int16")
        # convert to float 
        chunk = convert_audio_arr_to_float(chunk)

        #audio_size += len(chunk)

        for clip_path in clip_paths:
            clip_data = clip_datas[clip_path]

            clip = clip_data["clip"]
            clip_name = clip_data["clip_name"]
            clip_seconds = clip_data["clip_seconds"]
            sliding_window = clip_data["sliding_window"]

            peak_times = _find_clip_in_chunk(
                                            clip=clip, 
                                            clip_name=clip_name,
                                            clip_seconds=clip_seconds,
                                            index=i,
                                            previous_chunk=previous_chunk,
                                            chunk=chunk, 
                                            sliding_window=sliding_window,
                                            seconds_per_chunk=seconds_per_chunk, 
                                            method=method,
                                            correlation_threshold = correlation_threshold,
                                    )
    
            all_peak_times[clip_path].extend(peak_times)

        # Update previous_chunk to current chunk
        previous_chunk = chunk
        i = i + 1

    process.wait()

    return all_peak_times


def _find_clip_in_chunk(clip,clip_name,clip_seconds,index,previous_chunk,chunk,
                        sliding_window,seconds_per_chunk,method,correlation_threshold):

    unwind_clip_ts = True



    all_peak_times = []

    peak_times = process_chunk(chunk=chunk, clip=clip, sr=target_sample_rate,
                                previous_chunk=previous_chunk,
                                sliding_window=sliding_window,
                                index=index,
                                clip_name=clip_name,
                                seconds_per_chunk=seconds_per_chunk, method=method,
                                correlation_threshold = correlation_threshold
                                )
    if len(peak_times):
        peak_times_from_beginning = [time + (index * seconds_per_chunk) for time in peak_times]
        if unwind_clip_ts:
            peak_times_from_beginning_new = []
            for time in peak_times_from_beginning:
                new_time = time - clip_seconds
                if new_time >= 0:
                    peak_times_from_beginning_new.append(new_time)
                else:
                    peak_times_from_beginning_new.append(time)
            peak_times_from_beginning = peak_times_from_beginning_new
        #print(f"Found occurrences at: {peak_times} seconds, chunk {i}")
        all_peak_times.extend(peak_times_from_beginning)
        #all_correlation.extend(correlation)

    return all_peak_times

def set_debug_mode(debug):
    global debug_mode
    debug_mode = debug