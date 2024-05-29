import argparse
import sys
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
from scipy.signal import correlate, savgol_filter, find_peaks_cwt, peak_prominences, peak_widths
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


from peak_methods import get_peak_profile
from utils import is_unique_and_sorted

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


# def condense(values, factor):
#     x = np.arange(len(values))
#     y = values
#     x_condensed = x[::factor]
#     y_condensed = np.interp(x_condensed, x, y)  # Interpolate to smooth
#     return y_condensed
#     #return np.array([np.mean(values[i:i + factor]) for i in range(0, len(values), factor)])

def downsample(values,factor):
    buffer_ = deque([], maxlen=factor)
    downsampled_values = []
    for i, value in enumerate(values):
        buffer_.appendleft(value)
        if (i - 1) % factor == 0:
            # Take max value out of buffer
            # or you can take higher value if their difference is too big, otherwise just average
            max_value = max(buffer_)
            #if max_value > 0.2:
            downsampled_values.append(max_value)
            #else:
            #downsampled_values.append(np.mean(buffer_))
    return np.array(downsampled_values)

def max_distance(sorted_data):
    max_dist = 0
    for i in range(1, len(sorted_data)):
        dist = sorted_data[i] - sorted_data[i - 1]
        max_dist = max(max_dist, dist)
    return max_dist


# won't work well for very short clips like single beep
# because it gives too many false positives
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

    section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)

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
            f'{graph_dir}/{clip_name}_{index}_{section_ts}.png')
        plt.close()

    max_sample = len(audio) - samples_skip_end
    #trim placeholder clip
    correlation = correlation[:max_sample]

    if debug_mode:
        percentile = np.percentile(correlation, 99)
        max_correlation = max(correlation)
        ratio = percentile / max_correlation
        print(f"{section_ts} percentile", percentile)
        print(f"{section_ts} max correlation: {max_correlation}")
        print(f"{section_ts} ratio: {ratio}")
        print(f"---")

    #height = threshold
    distance = clip_length
    width = int(max(clip_length, 1 * sr) / 512)
    # find the peaks in the spectrogram
    peaks, properties = find_peaks(correlation, prominence=threshold, width=[0,width], distance=distance)


    if debug_mode:
        peak_dir = f"./tmp/peaks/cross_correlation_{clip_name}"
        os.makedirs(peak_dir, exist_ok=True)
        peaks_test=[]
        for i,item in enumerate(peaks):
            #plot_test_x=np.append(plot_test_x, index)
            #plot_test_y=np.append(plot_test_y, item)
            peaks_test.append([int(item),item/sr,correlation[item]])
        peaks_test.append({"properties":properties})
        print(json.dumps(peaks_test, indent=2,cls=NumpyEncoder), file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))

    #peaks = np.where(correlation > threshold)[0]

    peak_times = np.array(peaks) / sr

    #
    # max_time = ((len(audio) - 1) - samples_skip_end) / sr
    #
    # # Array operation to filter peak_times
    # peak_times2 = peak_times[(peak_times >= 0) & (peak_times <= max_time)]

    return peak_times

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_diff_ratio(control,value):
    diff = np.abs(control - value)
    #max_value = max(input1,input2)
    ratio = diff / control
    return ratio

def calculate_similarity(arr1, arr2):
  """Calculates the similarity between two normalized arrays
     using mean squared error.

  Args:
    arr1: The first normalized array.
    arr2: The second normalized array.

  Returns:
    A similarity score (lower is more similar) based on
    mean squared error.
  """
  return np.mean((arr1 - arr2)**2)

# won't work well if there are multiple occurrences of the same clip
# within the same audio_section because it inflates percentile
# and triggers multiple peaks elimination fallback
def non_repeating_correlation(clip, audio_section, sr, index, seconds_per_chunk, clip_name):
    # if clip_name == "日落大道interlude" and index not in [36,37,50,92]:
    #     return []
    # if clip_name == "日落大道smallinterlude" and index not in [13,14]:
    #     return []
    # if clip_name == "漫談法律intro" and index not in [10,11]:
    #     return []
    # if clip_name == "繼續有心人intro" and index not in [10,11]:
    #    return []
    # if clip_name == "rthk_news_report_theme" and index not in [26,27]:
    #    return []


    section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)

    clip_length = len(clip)

    downsample_factor = int(sr / 10)

    # Cross-correlate and normalize correlation
    correlation_clip = correlate(clip, clip, mode='full', method='fft')
    correlation_clip = downsample(correlation_clip, downsample_factor)

    # abs
    correlation_clip = np.abs(correlation_clip)
    correlation_clip /= np.max(correlation_clip)

    max_index_clip = np.argmax(correlation_clip)
    profile_clip = get_peak_profile(max_index_clip, correlation_clip)
    #bottom_ratio = get_diff_ratio(profile_clip["width_100"],profile_clip["width_75"])
    #profile_clip["bottom_ratio"] = bottom_ratio


    #correlation_clip=savgol_filter(correlation_clip, profile_clip["width_100"], 1)


    if debug_mode:
        print("clip_length", clip_length)
        print("correlation_clip_length", len(correlation_clip))
        graph_dir = f"./tmp/graph/clip_correlation"
        os.makedirs(graph_dir, exist_ok=True)

        plt.figure(figsize=(10, 4))

        plt.plot(correlation_clip)

        plt.title('Cross-correlation for clip')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coefficient')
        plt.savefig(
            f'{graph_dir}/{clip_name}.png')
        plt.close()

        print(f"{section_ts} prominence_width_clip",profile_clip["prominence"])
        #print(f"{section_ts} bottom_ratio",profile_clip["bottom_ratio"])
        #print(f"{section_ts} left_through",left_through)
        #print(f"{section_ts} right_through",right_through)
        #print(f"{section_ts} width_clip",profile["width_100"])
        print(f"{section_ts} width_clip_whole",profile_clip["width_100"])
        print(f"{section_ts} width_clip_75",profile_clip["width_75"])
        print(f"{section_ts} width_clip_half",profile_clip["width_50"])
        #print(f"{section_ts} width_middle",width_middle)
        #print(f"{section_ts} wlen", wlen)
        peak_dir = f"./tmp/peaks_clip/non_repeating_cross_correlation_{clip_name}"
        os.makedirs(peak_dir, exist_ok=True)
        print(json.dumps({"max_index":max_index_clip,
                          "profile":profile_clip
                          }, indent=2, cls=NumpyEncoder),
              file=open(f'{peak_dir}/{clip_name}.txt', 'w'))

    print("audio_section length",len(audio_section))
    # Cross-correlate and normalize correlation
    correlation = correlate(audio_section, clip, mode='full', method='fft')
    print("correlation length", len(correlation))

    # abs
    correlation = np.abs(correlation)
    correlation /= np.max(correlation)

    max_index = np.argmax(correlation)

    padding = clip_length

    beg = int(max_index-padding)
    end = int(max_index+padding)

    if beg < 0:
        end = end - beg
        beg = 0

    max_index_orig = np.argmax(correlation)

    if end >= len(correlation):
        correlation = np.pad(correlation, (0, end - len(correlation)), 'constant')
    # slice
    correlation = correlation[beg:end]


    correlation = downsample(correlation, downsample_factor)
    max_index_downsample = np.argmax(correlation)

    profile_section = get_peak_profile(max_index_downsample, correlation)

    diff_prominence_ratio = get_diff_ratio(profile_clip["prominence"],profile_section["prominence"])

    #print("correlation_clip_len_comp",len(correlation_clip), len(correlation))

    similarity = calculate_similarity(correlation_clip,correlation)

    if debug_mode:
        graph_dir = f"./tmp/graph/non_repeating_cross_correlation/{clip_name}"
        os.makedirs(graph_dir, exist_ok=True)

        #Optional: plot the correlation graph to visualize
        plt.figure(figsize=(10, 4))
        # if clip_name == "漫談法律intro" and index == 10:
        #     plt.plot(correlation[454000:454100])
        # elif clip_name == "漫談法律intro" and index == 11:
        #     plt.plot(correlation[50000:70000])
        # elif clip_name == "日落大道smallinterlude" and index == 13:
        #     plt.plot(correlation[244100:244700])
        # elif clip_name == "日落大道smallinterlude" and index == 14:
        #     plt.plot(correlation[28300:28900])
        # elif clip_name == "繼續有心人intro" and index == 10:
        #     plt.plot(correlation[440900:441000])
        # else:
        #     plt.plot(correlation)
        plt.plot(correlation)

        plt.title('Cross-correlation between the audio clip and full track before slicing')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coefficient')
        plt.savefig(
            f'{graph_dir}/{clip_name}_{index}_{section_ts}.png')
        plt.close()

        print(f"{section_ts} similarity",similarity)
        print(f"{section_ts} prominence",profile_section["prominence"])
        #print(f"{section_ts} bottom_ratio",profile_section["bottom_ratio"])
        #print(f"{section_ts} diff_bottom_ratio",diff_bottom_ratio)
        print(f"{section_ts} width_100",profile_section["width_100"])
        print(f"{section_ts} width_75",profile_section["width_75"])
        print(f"{section_ts} width_half",profile_section["width_50"])

        #print(f"{section_ts} diff_prominence_ratio",diff_prominence_ratio)
        #print(f"{section_ts} diff_width_100_ratio",diff_width_100_ratio)
        #print(f"{section_ts} diff_width_75_ratio",diff_width_75_ratio)
        #print(f"{section_ts} diff_width_50_ratio",diff_width_50_ratio)

        peak_dir = f"./tmp/peaks/non_repeating_cross_correlation_{clip_name}"
        os.makedirs(peak_dir, exist_ok=True)
        print(json.dumps({"max_index":max_index_downsample,
                          "profile":profile_section
                          }, indent=2, cls=NumpyEncoder),
              file=open(f'{peak_dir}/{clip_name}_{index}_{section_ts}.txt', 'w'))

    qualified = True
    if diff_prominence_ratio > 0.1:
        print(f"failed verification for {section_ts} due to prominence ratio {diff_prominence_ratio}")
        qualified = False
    if similarity > 0.01:
        print(f"failed verification for {section_ts} due to similarity {similarity}")
        qualified = False
    #if diff_width_100_ratio > 0.1:
    #    print(f"failed verification for {section_ts} due to width_100 ratio {diff_width_100_ratio}")
    #    qualified = False
    #if diff_bottom_ratio > 0.2:
    #    print(f"failed verification for {section_ts} due to width_75diff_bottom_ratio ratio {diff_bottom_ratio}")
    #    qualified = False
    #if diff_width_75_ratio > 0.5:
    #    print(f"failed verification for {section_ts} due to width_75 ratio {diff_width_75_ratio}")
    #    qualified = False
    #if diff_width_50_ratio > 0.1:
    #    print(f"failed verification for {section_ts} due to width_50 ratio {diff_width_50_ratio}")
    #    qualified = False

    if not qualified:
        print(f"failed verification for {section_ts}")
        return []
    else:
        return [max_index_orig / sr]




# sliding_window: for previous_chunk in seconds from end
# index: for debugging by saving a file for audio_section
# seconds_per_chunk: default seconds_per_chunk
def process_chunk(chunk, clip, sr, previous_chunk, sliding_window, index, seconds_per_chunk, clip_name,
                  method, threshold=None):
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
            audio_section = np.concatenate((previous_chunk[int(-sliding_window * sr):], chunk, np.array([])))
    else:
        subtract_seconds = 0
        audio_section = np.concatenate((chunk, np.array([])))

    normalize = True

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

    os.makedirs("./tmp/audio", exist_ok=True)
    if debug_mode:
        sf.write(
            f"./tmp/audio/section_{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.wav",
            audio_section, sr)

    if method == "correlation":
        if threshold is None:
            raise ValueError("threshold is required for correlation method")
        # samples_skip_end does not skip results from being included yet
        peak_times = correlation_method(clip, audio_section=audio_section, sr=sr, index=index,
                                        seconds_per_chunk=seconds_per_chunk,
                                        clip_name=clip_name,
                                        threshold=threshold)
    elif method == "non_repeating_correlation":
    #    peak_times = correlation_method(clip, audio_section=audio_section, sr=sr, index=index,
    #                                    seconds_per_chunk=seconds_per_chunk,
    #                                    clip_name=clip_name,
    #                                    threshold=threshold,repeating=False)
    #elif method == "experimental_non_repeating_correlation":
        peak_times = non_repeating_correlation(clip, audio_section=audio_section, sr=sr, index=index,
                                        seconds_per_chunk=seconds_per_chunk, clip_name=clip_name)
        # peak_times=[]
        # if(len(peak_times_tentative)<=1):
        #     peak_times=peak_times_tentative
        # else:
        #     if not is_unique_and_sorted(peak_times_tentative):
        #         raise ValueError(f"peak_times_tentative is not unique and sorted {peak_times_tentative}, maybe clean up first before selecting good ones")
        #     peak_times_tentative.append(sys.maxsize)
        #     for i in range(1,len(peak_times_tentative)):
        #         last_one = i == len(peak_times_tentative) - 1
        #         peak_time = peak_times_tentative[i]
        #         prev_peak_time = peak_times_tentative[i-1]
        #         #if peak_time is not None:
        #         #    continue
        #         if peak_time - prev_peak_time < seconds_per_chunk:
        #             print(f"skipping {prev_peak_time} due to less than {seconds_per_chunk} has match prev")
        #             continue
        #         if not last_one:
        #             next_peak_time = peak_times_tentative[i + 1]
        #             if next_peak_time - peak_time < seconds_per_chunk:
        #                 print(f"skipping {peak_time} due to less than {seconds_per_chunk} has match next")
        #                 continue
        #         peak_times.append(prev_peak_time)
    else:
        raise ValueError("unknown method")

    peak_times_final = [peak_time - subtract_seconds for peak_time in peak_times]

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
    sliding_window = math.ceil(clip_seconds)

    if (sliding_window != clip_seconds):
        print(f"adjusted sliding_window from {clip_seconds} to {sliding_window} for {clip_name}")
    # sliding_window = 5
    #
    # if (sliding_window < clip_seconds + 5):
    #     # need to extend the sliding window to overlap the clip
    #     sliding_window = clip_seconds + 5
    #     print(f"adjusted sliding_window to {sliding_window} for {clip_name}")

    # this should not happen anyways because the seconds per chunk is too small
    if (seconds_per_chunk < sliding_window * 2):
        seconds_per_chunk = sliding_window * 10
        raise ValueError(f"seconds_per_chunk {seconds_per_chunk} is too small")

    return sliding_window

# could cause issues with small overlap when intro is followed right by news report
def find_clip_in_audio_in_chunks(clip_paths, full_audio_path, method, threshold=None):
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
                                            threshold=threshold,
                                    )
    
            all_peak_times[clip_path].extend(peak_times)

        # Update previous_chunk to current chunk
        previous_chunk = chunk
        i = i + 1

    process.wait()

    return all_peak_times


def _find_clip_in_chunk(clip, clip_name, clip_seconds, index, previous_chunk, chunk,
                        sliding_window, seconds_per_chunk, method, threshold):

    unwind_clip_ts = True



    all_peak_times = []

    peak_times = process_chunk(chunk=chunk, clip=clip, sr=target_sample_rate,
                               previous_chunk=previous_chunk,
                               sliding_window=sliding_window,
                               index=index,
                               clip_name=clip_name,
                               seconds_per_chunk=seconds_per_chunk, method=method,
                               threshold=threshold
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