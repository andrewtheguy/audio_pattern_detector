import argparse
import sys
from collections import deque, defaultdict
import copy
import datetime
import json
import logging
import os
import pdb
import time
from operator import itemgetter
from pathlib import Path

import dtaidistance
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from numpy_encoder import NumpyEncoder
from peak_methods import get_peak_profile, find_closest_troughs
from utils import slicing_with_zero_padding, downsample, area_of_overlap_ratio

logger = logging.getLogger(__name__)

#ignore possible clipping
warnings.filterwarnings('ignore', module='pyloudnorm')

DEFAULT_METHOD="correlation"

plot_test_x = np.array([])
plot_test_y = np.array([])

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

def convert_audio_arr_to_float(audio):
    #raise "chafa"
    return librosa.util.buf_to_float(audio,n_bytes=2, dtype='float32')



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

# def dtw_distance(series1, series2):
#     distance, path = fastdtw(series1, series2, dist=2)
#     return distance
#     #return d


def stretch_target(reference, target, ref_peak_info, target_peak_info):
    ref_left_trough, ref_peak, ref_right_trough = ref_peak_info
    target_left_trough, target_peak, target_right_trough = target_peak_info

    # Segments to stretch
    segments = [
        (0, target_left_trough),
        (target_left_trough, target_peak),
        (target_peak, target_right_trough),
        (target_right_trough, len(target) - 1)
    ]

    # Corresponding reference segment lengths
    ref_lengths = [
        ref_left_trough,
        ref_peak - ref_left_trough,
        ref_right_trough - ref_peak,
        len(reference) - 1 - ref_right_trough
    ]

    stretched_target = []

    # Apply proportional stretching to each segment
    for (start, end), ref_length in zip(segments, ref_lengths):
        target_segment = target[start:end + 1]
        if len(target_segment) > 1:
            target_indices = np.linspace(start, end, len(target_segment))
            stretched_indices = np.linspace(0, ref_length, len(target_segment))
            stretched_segment = np.interp(stretched_indices, np.arange(ref_length + 1), target_segment)
            stretched_target.extend(stretched_segment)
        else:
            stretched_target.append(target[start])

    return np.array(stretched_target)

# Normalize the curves to the same scale
def normalize_curve(curve):
    return (curve - np.min(curve)) / (np.max(curve) - np.min(curve))

# Apply DTW and warp the target curve
def warp_with_dtw(reference, target):
    # Compute dynamic time warping path
    path = dtaidistance.dtw.warping_path(reference, target)
    #print("path",path)

    # Create an array to hold the warped target
    warped_target = np.zeros_like(reference)
    for ref_idx, target_idx in path:
        warped_target[ref_idx] = target[target_idx]

    return warped_target, path

def downsample_preserve_maxima(curve, num_samples):
    n_points = len(curve)
    step_size = n_points / num_samples
    compressed_curve = []

    for i in range(num_samples):
        start_index = int(i * step_size)
        end_index = int((i + 1) * step_size)

        if start_index >= n_points:
            break

        window = curve[start_index:end_index]
        if len(window) == 0:
            continue

        local_max_index = np.argmax(window)
        compressed_curve.append(window[local_max_index])

        # # Find peaks within this window
        # peaks, _ = find_peaks(window)
        # if len(peaks) == 0:
        #     # If no peaks, simply downsample by taking the first point in the window
        #     compressed_curve.append(window[0])
        # else:
        #     # Select the local maxima point
        #     local_max_index = peaks[np.argmax(window[peaks])]
        #     compressed_curve.append(window[local_max_index])

    # Adjust the length if necessary by adding the last element of the original curve
    if len(compressed_curve) < num_samples and len(curve) > 0:
        compressed_curve.append(curve[-1])

    if len(compressed_curve) != num_samples:
        raise ValueError(f"downsampled curve length {len(compressed_curve)} not equal to num_samples {num_samples}")

    return np.array(compressed_curve)

class AudioOffsetFinder:
    def __init__(self, clip_paths, method=DEFAULT_METHOD,debug_mode=False):
        self.clip_paths = clip_paths
        self.method = method
        self.debug_mode = debug_mode
        #self.correlation_cache_correlation_method = {}
        self.normalize = True
        self.target_sample_rate = 8000
        self.target_num_sample_after_resample = 101
        self.similarity_debug=defaultdict(list)
        self.areas_debug=defaultdict(list)
        self.similarity_method = "mse"
        if self.similarity_method == "mse":
            self.similarity_threshold = 0.002
            # for very short clip
            #self.area_threshold=0.085
            #self.very_short_clip_similarity_threshold = 0.01
            ## lower threshold for conditional check on very short clip because the higher likelihood of false positives
            ## check for area difference
            #self.very_short_clip_similarity_threshold_conditional = 0.003
        elif self.similarity_method == "mae": #median_absolute_error, a bit better for news report beep
            self.similarity_threshold = 0.02
        else:
            raise ValueError("unknown similarity method")

    # could cause issues with small overlap when intro is followed right by news report
    def find_clip_in_audio(self, full_audio_path):
        clip_paths = self.clip_paths
        #self.correlation_cache_correlation_method.clear()
        for clip_path in clip_paths:
            if not os.path.exists(clip_path):
                raise ValueError(f"Clip {clip_path} does not exist")

        if not os.path.exists(full_audio_path):
            raise ValueError(f"Full audio {full_audio_path} does not exist")

        seconds_per_chunk = 60

        # 2 bytes per channel on every sample for 16 bits (int16)
        # times two because it is (int16, mono)
        chunk_size = (seconds_per_chunk * self.target_sample_rate) * 2

        # Initialize parameters

        previous_chunk = None  # Buffer to maintain continuity between chunks

        all_peak_times = {clip_path: [] for clip_path in clip_paths}


        # Create ffmpeg process
        process = (
            ffmpeg
            .input(full_audio_path)
            .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=self.target_sample_rate, loglevel="error")
            .run_async(pipe_stdout=True)
        )
        #audio_size = 0

        i = 0

        clip_datas={}

        for clip_path in clip_paths:
            # Load the audio clip
            clip = load_audio_file(clip_path, sr=self.target_sample_rate)
            # convert to float
            clip = convert_audio_arr_to_float(clip)

            clip_name, _ = os.path.splitext(os.path.basename(clip_path))

            clip_seconds = len(clip) / self.target_sample_rate

            sliding_window = get_chunking_timing_info(clip_name,clip_seconds,seconds_per_chunk)

            if self.normalize:
                clip_length = len(clip)
                sr = self.target_sample_rate
                #clip_second = clip_length / sr

                # normalize loudness
                if clip_seconds < 0.5:
                    meter = pyln.Meter(sr, block_size=clip_seconds)
                else:
                    meter = pyln.Meter(sr)  # create BS.1770 meter
                loudness = meter.integrated_loudness(clip)

                # loudness normalize audio to -12 dB LUFS
                clip = pyln.normalize.loudness(clip, loudness, -12.0)

            correlation_clip = self._get_clip_correlation(clip, clip_name)

            if self.debug_mode:
                print(f"clip_length {clip_name}", len(clip))
                print(f"clip_length {clip_name} seconds", len(clip)/self.target_sample_rate)
                print("correlation_clip_length", len(correlation_clip))
                graph_dir = f"./tmp/graph/clip_correlation"
                os.makedirs(graph_dir, exist_ok=True)

                plt.figure(figsize=(10, 4))

                plt.plot(correlation_clip)
                plt.title('Cross-correlation of the audio clip itself')
                plt.xlabel('Lag')
                plt.ylabel('Correlation coefficient')
                plt.savefig(
                    f'{graph_dir}/{clip_name}.png')
                plt.close()

            #downsampled_correlation_clip = downsample_preserve_maxima(correlation_clip, self.target_num_sample_after_resample)
            downsampled_correlation_clip = downsample(correlation_clip, len(correlation_clip)//500)

            if self.debug_mode:
                print("downsampled_correlation_clip_length", len(downsampled_correlation_clip))
                graph_dir = f"./tmp/graph/clip_correlation_downsampled"
                os.makedirs(graph_dir, exist_ok=True)

                plt.figure(figsize=(10, 4))

                plt.plot(downsampled_correlation_clip)
                plt.title('Cross-correlation of the audio clip itself')
                plt.xlabel('Lag')
                plt.ylabel('Correlation coefficient')
                plt.savefig(
                    f'{graph_dir}/{clip_name}.png')
                plt.close()

            clip_datas[clip_path] = {"clip":clip,
                                     "clip_name":clip_name,
                                     "sliding_window":sliding_window,
                                     "correlation_clip":correlation_clip,
                                     "downsampled_correlation_clip":downsampled_correlation_clip,
                                     }



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


                peak_times = self._process_chunk(chunk=chunk,
                                                 sr=self.target_sample_rate,
                                                 previous_chunk=previous_chunk,
                                                 index=i,
                                                 clip_data=clip_data,
                                                 seconds_per_chunk=seconds_per_chunk,
                                                 )

                all_peak_times[clip_path].extend(peak_times)

            # Update previous_chunk to current chunk
            previous_chunk = chunk
            i = i + 1

        suffix = Path(full_audio_path).stem

        if self.debug_mode and (self.method == "correlation" or self.method == "non_repeating_correlation"):
            for clip_path in clip_paths:
                clip_name, _ = os.path.splitext(os.path.basename(clip_path))

                # similarity debug
                graph_dir = f"./tmp/graph/{self.method}_similarity_{self.similarity_method}/{clip_name}"
                os.makedirs(graph_dir, exist_ok=True)

                x_coords = []
                y_coords = []

                for index,arr in enumerate(self.similarity_debug[clip_name]):
                    for item in arr:
                        #if item <= 0.02:
                        x_coords.append(index)
                        y_coords.append(item)

                plt.figure(figsize=(10, 4))
                # Create scatter plot
                plt.scatter(x_coords, y_coords)

                # Adding titles and labels
                plt.title('Scatter Plot for Similarity')
                plt.xlabel('Value')
                plt.ylabel('Sublist Index')
                plt.savefig(
                    f'{graph_dir}/{suffix}.png')
                plt.close()

                # ares debug
                # graph_dir = f"./tmp/graph/{self.method}_area_{self.similarity_method}/{clip_name}"
                # os.makedirs(graph_dir, exist_ok=True)
                #
                # x_coords = []
                # y_coords = []
                #
                # for index,arr in enumerate(self.areas_debug[clip_name]):
                #     for item in arr:
                #         #if item <= 0.02:
                #         x_coords.append(index)
                #         y_coords.append(item)
                #
                # plt.figure(figsize=(10, 4))
                # # Create scatter plot
                # plt.scatter(x_coords, y_coords)
                #
                # # Adding titles and labels
                # plt.title('Scatter Plot for Areas')
                # plt.xlabel('Value')
                # plt.ylabel('Sublist Index')
                # plt.savefig(
                #     f'{graph_dir}/{suffix}.png')
                # plt.close()

        process.wait()

        return all_peak_times

    def _get_clip_correlation(self, clip, clip_name):
        # Cross-correlate and normalize correlation
        correlation_clip = correlate(clip, clip, mode='full', method='fft')

        # abs
        correlation_clip = np.abs(correlation_clip)
        correlation_clip /= np.max(correlation_clip)

        return correlation_clip


    # sliding_window: for previous_chunk in seconds from end
    # index: for debugging by saving a file for audio_section
    # seconds_per_chunk: default seconds_per_chunk
    def _process_chunk(self, chunk, clip_data, sr, previous_chunk, index, seconds_per_chunk):
        debug_mode = self.debug_mode
        clip, clip_name, sliding_window = itemgetter("clip","clip_name","sliding_window")(clip_data)
        clip_length = len(clip)
        clip_seconds = len(clip) / sr
        chunk_seconds = len(chunk) / sr
        # Concatenate previous chunk for continuity in processing
        if previous_chunk is not None:
            if chunk_seconds < seconds_per_chunk:  # too small
                # no need for sliding window since it is the last piece
                subtract_seconds = -(chunk_seconds - seconds_per_chunk)
                audio_section_temp = np.concatenate((previous_chunk, chunk))[(-seconds_per_chunk * sr):]
                audio_section = np.concatenate((audio_section_temp, np.array([])))
            else:
                subtract_seconds = sliding_window
                audio_section = np.concatenate((previous_chunk[int(-sliding_window * sr):], chunk, np.array([])))
        else:
            subtract_seconds = 0
            audio_section = np.concatenate((chunk, np.array([])))

        if self.normalize:
            audio_section_seconds = len(audio_section) / sr
            #normalize loudness
            if audio_section_seconds < 0.5:
                meter = pyln.Meter(sr, block_size=audio_section_seconds)
            else:
                meter = pyln.Meter(sr)  # create BS.1770 meter

            loudness = meter.integrated_loudness(audio_section)

            # loudness normalize audio to -12 dB LUFS
            audio_section = pyln.normalize.loudness(audio_section, loudness, -12.0)


        # if debug_mode:
        #     os.makedirs("./tmp/audio", exist_ok=True)
        #     sf.write(
        #         f"./tmp/audio/section_{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.wav",
        #         audio_section, sr)

        if self.method == "correlation":

            # samples_skip_end does not skip results from being included yet
            peak_times = self._correlation_method(clip_data, audio_section=audio_section, sr=sr, index=index,
                                                  seconds_per_chunk=seconds_per_chunk, one_shot=False

                                                  )
        elif self.method == "non_repeating_correlation":
            peak_times = self._correlation_method(clip_data, audio_section=audio_section, sr=sr, index=index,
                                                  seconds_per_chunk=seconds_per_chunk, one_shot=True
                                                  )

        else:
            raise ValueError("unknown method")

        # subtract sliding window seconds from peak times
        peak_times = [peak_time - subtract_seconds for peak_time in peak_times]

        # move timestamp to be before the clip
        if len(peak_times):
            peak_times_from_beginning = [time + (index * seconds_per_chunk) for time in peak_times]
            peak_times_from_beginning_new = []
            for time in peak_times_from_beginning:
                new_time = time - clip_seconds
                if new_time >= 0:
                    peak_times_from_beginning_new.append(new_time)
                else:
                    peak_times_from_beginning_new.append(0)
            peak_times_final = peak_times_from_beginning_new
        else:
            peak_times_final = []

        return peak_times_final

    def _calculate_similarity(self, correlation_clip, correlation_slice):

        if self.similarity_method == "mae":
            similarity = median_absolute_error(correlation_clip, correlation_slice)
        elif self.similarity_method == "mse":
            similarity = mean_squared_error(correlation_clip, correlation_slice)
        else:
            raise ValueError("unknown similarity method")
        return similarity

    def _calculate_area_of_overlap_ratio(self, correlation_clip, correlation_slice, downsampled_correlation_clip):
        downsampled_correlation_slice = downsample(correlation_slice, len(correlation_clip)//500)

        #
        # peak_index = np.argmax(downsampled_correlation_clip)
        # peak_index_slice = np.argmax(downsampled_correlation_slice)
        # if self.debug_mode:
        #     print("len", len(downsampled_correlation_clip))
        #     print("peak_index", peak_index)
        # #raise "chafa"
        # if(peak_index != peak_index_slice):
        #     logger.warning(f"peak {peak_index_slice} not aligned with the original clip {peak_index}, potential bug in the middle of the chain")
        # left_trough, right_trough = find_closest_troughs(peak_index, downsampled_correlation_clip)
        # max_width_half = max(peak_index-left_trough,right_trough-peak_index)
        #
        # if max_width_half < 10:
        #     max_width_half = 10
        #
        # new_left = max(0,peak_index-max_width_half)
        # new_right = min(len(downsampled_correlation_clip),peak_index+max_width_half+1)
        #
        # clip_within_peak = downsampled_correlation_clip[new_left:new_right]
        # correlation_slice_within_peak = downsampled_correlation_slice[new_left:new_right]

        # alternative
        # peaks, properties = find_peaks(downsampled_correlation_clip, height=0.95,prominence=0.25,distance=21,wlen=21)
        # if len(peaks) != 1:
        #     raise ValueError(f"expected 1 peak, found {peaks}")
        #
        # new_left = properties["left_bases"][0]
        # new_right = properties["right_bases"][0]
        #
        # clip_within_peak = downsampled_correlation_clip[new_left:new_right]
        # correlation_slice_within_peak = downsampled_correlation_slice[new_left:new_right]

        # static
        middle = len(downsampled_correlation_clip) // 2

        # mid point of the peak for 1/10th of the downsampled_correlation_clip
        new_left = middle - len(downsampled_correlation_clip) // 20
        new_right = middle + len(downsampled_correlation_clip) // 20

        clip_within_peak = downsampled_correlation_clip[new_left:new_right]
        correlation_slice_within_peak = downsampled_correlation_slice[new_left:new_right]

        # clip the tails
        area_of_overlap, props = area_of_overlap_ratio(clip_within_peak,
                                                       correlation_slice_within_peak)

        return area_of_overlap,{
            #"clip_within_peak":clip_within_peak,
            #"correlation_slice_within_peak":correlation_slice_within_peak,
            "downsampled_correlation_slice":downsampled_correlation_slice,
            "new_left":new_left,
            "new_right":new_right,
            "area_props":props,
        }

    # one_shot = True will only check the max height,
    # faster than repeat method and picks up a potential very soft one
    # won't work well if there are multiple occurrences of the same clip
    # because it only picks the best one
    # won't work well for very short clips like single beep
    # because it is more likely to have false positives or miss good ones
    def _correlation_method(self, clip_data, audio_section, sr, index, seconds_per_chunk, one_shot=False):
        clip, clip_name, sliding_window, correlation_clip, downsampled_correlation_clip = (
            itemgetter("clip","clip_name","sliding_window","correlation_clip","downsampled_correlation_clip")(clip_data))
        debug_mode = self.debug_mode

        clip_length = len(clip)

        #very_short_clip = len(clip) < 0.75 * sr

        if not one_shot: #eliminate obviously bad ones by comparing with original clip and select by certain threshold
            zeroes_second_pad = 1
            # pad zeros between audio and clip
            zeroes = np.zeros(clip_length + zeroes_second_pad * sr)
            audio = np.concatenate((audio_section, zeroes, clip))
            samples_skip_end = zeroes_second_pad * sr + clip_length
        else:
            audio = audio_section
            samples_skip_end = 0

        # Cross-correlate and normalize correlation
        correlation = correlate(audio, clip, mode='full', method='fft')
        # abs
        correlation = np.abs(correlation)
        # alternative to replace negative values with zero in array instead of above
        #correlation[correlation < 0] = 0
        correlation /= np.max(correlation)

        section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)

        if debug_mode:
            print(f"---")
            print(f"section_ts: {section_ts}")
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

        if not one_shot:
            distance = clip_length
            height_max = 0.25
            peaks, _ = find_peaks(correlation, height=height_max, distance=distance)
        else:
            peaks = [np.argmax(correlation)]

        peaks_final = []

        # for debugging
        similarities = []
        correlation_slices = []

        for peak in peaks:
            after = peak + len(correlation_clip)//2
            before = peak - len(correlation_clip)//2
            if after > len(correlation)-1+2:
                logger.warning(f"peak {peak} after is {after} > len(correlation)+2 {len(correlation)+2}, skipping")
                similarities.append((1,1,1,1,))
                correlation_slices.append([])
                continue
            elif before < -2:
                logger.warning(f"peak {peak} before is {before} < -2, skipping")
                similarities.append((1,1,1,1,))
                correlation_slices.append([])
                continue


            # slice
            correlation_slice = slicing_with_zero_padding(correlation, len(correlation_clip), peak)
            correlation_slice = correlation_slice/np.max(correlation_slice)

            if len(correlation_slice) != len(correlation_clip):
                raise ValueError(f"correlation_slice length {len(correlation_slice)} not equal to correlation_clip length {len(correlation_clip)}")

            quarter = len(correlation_clip) // 4

            #if np.argmax(correlation_slice) != np.argmax(correlation_clip):
            #    raise ValueError(f"peak {np.argmax(correlation_slice)} not aligned with the original clip {np.argmax(correlation_clip)}, potential bug in the middle of the chain")

            similarity_quadrants = []
            for i in range(4):
                similarity_quadrants.append(self._calculate_similarity(correlation_slice=correlation_slice[i*quarter:(i+1)*quarter],
                                                                      correlation_clip=correlation_clip[i*quarter:(i+1)*quarter]))

            similarity_left = (similarity_quadrants[0]+similarity_quadrants[1])/2
            similarity_middle = (similarity_quadrants[1]+similarity_quadrants[2])/2
            similarity_right = (similarity_quadrants[2]+similarity_quadrants[3])/2


            similarity = min(similarity_left,similarity_middle,similarity_right)

            if debug_mode:
                print("similarity", similarity)

                similarity_whole = (similarity_left + similarity_right) / 2

                #if similarity <= 0.01:
                similarities.append((similarity,similarity_whole,similarity_left,similarity_middle,similarity_right,))
                correlation_slices.append(correlation_slice)

            # if very_short_clip:
            #     if similarity > self.very_short_clip_similarity_threshold:
            #         if debug_mode:
            #             print(f"failed verification for {section_ts} due to similarity {similarity} > {self.very_short_clip_similarity_threshold}")
            #     elif self.similarity_method == "mse" and similarity > self.very_short_clip_similarity_threshold_conditional:
            #         area_ratio,_ = self._calculate_area_of_overlap_ratio(correlation_clip,
            #                                                                 correlation_slice,
            #                                                              downsampled_correlation_clip,)
            #
            #         if area_ratio < self.area_threshold:
            #             peaks_final.append(peak)
            #         else:
            #             if debug_mode:
            #                 print(f"failed verification for very short clip {section_ts} due to area_of_overlap {area_ratio} >= {self.area_threshold}")
            #     else:
            #         peaks_final.append(peak)
            # else:
            if similarity > self.similarity_threshold:
                if debug_mode:
                    print(f"failed verification for {section_ts} due to similarity {similarity} > {self.similarity_threshold}")
            else:
                peaks_final.append(peak)

        if debug_mode:
            filtered_similarities = []
            for i,peak in enumerate(peaks):
                similarity = similarities[i][0]
                correlation_slice = correlation_slices[i]
                graph_max = 0.01
                if similarity <= graph_max:
                    filtered_similarities.append(similarity)
                    graph_dir = f"./tmp/graph/cross_correlation_slice/{clip_name}"
                    os.makedirs(graph_dir, exist_ok=True)

                    # Optional: plot the correlation graph to visualize
                    plt.figure(figsize=(10, 4))
                    plt.plot(correlation_slice)
                    plt.plot(correlation_clip)
                    plt.title('Cross-correlation between the audio clip and full track before slicing')
                    plt.xlabel('Lag')
                    plt.ylabel('Correlation coefficient')
                    plt.savefig(
                        f'{graph_dir}/{clip_name}_{index}_{section_ts}_{peak}.png')
                    plt.close()

            if len(filtered_similarities) > 0:
                peak_dir = f"./tmp/debug/cross_correlation_{clip_name}"
                os.makedirs(peak_dir, exist_ok=True)
                seconds=[]
                #distances = []

                area_props=[]
                for i,item in enumerate(peaks):
                    seconds.append(item / sr)
                #    correlation_slice = correlation_slices[i]
                #
                #
                #     area_ratio,props = self._calculate_area_of_overlap_ratio(correlation_clip,
                #                                                             correlation_slice,
                #                                                              downsampled_correlation_clip)
                #
                #     #clip_within_peak = props["clip_within_peak"]
                #     #correlation_slice_within_peak = props["correlation_slice_within_peak"]
                #
                #     downsampled_correlation_slice = props["downsampled_correlation_slice"]
                #     area_props.append(props["area_props"])
                #     new_left = props["new_left"]
                #     new_right = props["new_right"]
                #
                #
                #     graph_dir = f"./tmp/graph/cross_correlation_slice_downsampled/{clip_name}"
                #     os.makedirs(graph_dir, exist_ok=True)
                #
                #     plt.figure(figsize=(10, 4))
                #     plt.plot(downsampled_correlation_slice)
                #     plt.plot(downsampled_correlation_clip)
                #     plt.title('Cross-correlation between the audio clip and full track after slicing')
                #     plt.xlabel('Lag')
                #     plt.ylabel('Correlation coefficient')
                #     plt.savefig(
                #         f'{graph_dir}/{clip_name}_{index}_{section_ts}_{item}.png')
                #     plt.close()
                #
                #     areas.append(area_ratio)

                print(json.dumps({"peaks":peaks,"seconds":seconds,
                                  #"areas":areas,
                                  "area_props":area_props,
                                  #"pdc":pdc,
                                  #"peak_profiles":peak_profiles,
                                  #"pds":pds,
                                  #"properties":properties,
                                  #"new_left":new_left,
                                  #"new_right":new_right,
                                  "similarities":similarities}, indent=2,cls=NumpyEncoder), file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))
            self.similarity_debug[clip_name].append(filtered_similarities)
            #self.areas_debug[clip_name].append(areas)

            print(f"---")

        peak_times = np.array(peaks_final) / sr

        return peak_times


