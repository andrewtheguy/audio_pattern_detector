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
        self.similarity_debug=defaultdict(list)
        self.similarity_method = "mse"
        if self.similarity_method == "mse":
            self.similarity_threshold = 0.005
            # for very short clip
            self.very_short_clip_similarity_threshold = 0.01
            # lower threshold for conditional check on very short clip because the higher likelihood of false positives
            # check for area difference
            self.very_short_clip_similarity_threshold_conditional = 0.003
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



            # if self.debug_mode:
            #     print("downsampled_correlation_clip_length", len(downsampled_correlation_clip))
            #     graph_dir = f"./tmp/graph/clip_correlation_downsampled"
            #     os.makedirs(graph_dir, exist_ok=True)
            #
            #     plt.figure(figsize=(10, 4))
            #
            #     plt.plot(downsampled_correlation_clip)
            #     plt.title('Cross-correlation of the audio clip itself')
            #     plt.xlabel('Lag')
            #     plt.ylabel('Correlation coefficient')
            #     plt.savefig(
            #         f'{graph_dir}/{clip_name}.png')
            #     plt.close()

            clip_datas[clip_path] = {"clip":clip,
                                     "clip_name":clip_name,
                                     "sliding_window":sliding_window,
                                     "correlation_clip":correlation_clip
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

        if self.debug_mode and (self.method == "correlation" or self.method == "non_repeating_correlation"):
            for clip_path in clip_paths:
                clip_name, _ = os.path.splitext(os.path.basename(clip_path))
                #print("self.similarity_debug[clip_name]",self.similarity_debug[clip_name])
                graph_dir = f"./tmp/graph/{self.method}_similarity_{self.similarity_method}"
                os.makedirs(graph_dir, exist_ok=True)

                x_coords = []
                y_coords = []

                for index,arr in enumerate(self.similarity_debug[clip_name]):
                    for item in arr:
                        #if item <= 0.02:
                        x_coords.append(index)
                        y_coords.append(item)


                # Optional: plot the correlation graph to visualize
                plt.figure(figsize=(10, 4))
                # Create scatter plot
                plt.scatter(x_coords, y_coords)

                # Adding titles and labels
                plt.title('Scatter Plot for Given Array')
                plt.xlabel('Value')
                plt.ylabel('Sublist Index')
                plt.savefig(
                    f'{graph_dir}/{clip_name}.png')
                plt.close()

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
                                            seconds_per_chunk=seconds_per_chunk,repeating=True

                                                  )
        elif self.method == "non_repeating_correlation":
            peak_times = self._correlation_method(clip_data, audio_section=audio_section, sr=sr, index=index,
                                                        seconds_per_chunk=seconds_per_chunk,repeating=False
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

    def _calculate_area_of_overlap_ratio(self, correlation_clip, correlation_slice):
        downsampling_factor = 101
        downsampled_correlation_clip = downsample_preserve_maxima(correlation_clip, downsampling_factor)
        downsampled_correlation_slice = downsample_preserve_maxima(correlation_slice, downsampling_factor)
        peak_index = np.argmax(downsampled_correlation_clip)
        peak_index_slice = np.argmax(downsampled_correlation_slice)
        if self.debug_mode:
            print("len", len(downsampled_correlation_clip))
            print("peak_index", peak_index)
        #raise "chafa"
        if(peak_index != peak_index_slice):
            logger.warning(f"peak {peak_index_slice} not aligned with the original clip {peak_index}, potential bug in the middle of the chain")
        left_trough, right_trough = find_closest_troughs(peak_index, downsampled_correlation_clip)
        max_width = max(peak_index-left_trough,right_trough-peak_index)

        if max_width < 10:
            max_width = 10

        #scale = downsampled_correlation_clip[left_trough] / downsampled_correlation_slice[left_trough]

        # # Normalize the curves
        # reference_curve_norm = normalize_curve(downsampled_correlation_clip)
        # target_curve_norm = normalize_curve(downsampled_correlation_slice)
        #
        # # Apply DTW to warp the target curve to align with the reference curve
        # downsampled_correlation_slice, path = warp_with_dtw(reference_curve_norm, target_curve_norm)

        # target_left_trough, target_right_trough = find_closest_troughs(peak_index_slice, downsampled_correlation_slice)
        #
        # # Stretch target curve to match the reference curve
        # stretched_target_curve = stretch_target(downsampled_correlation_clip, downsampled_correlation_slice,
        #                                         (left_trough, peak_index, right_trough),
        #                                         (left_trough, peak_index_slice, right_trough))

        #
        new_left = max(0,peak_index-max_width)
        new_right = min(len(downsampled_correlation_clip),peak_index+max_width+1)
        #
        clip_within_peak = downsampled_correlation_clip[new_left:new_right]
        correlation_slice_within_peak = downsampled_correlation_slice[new_left:new_right]

        # downsampled_correlation_clip_peak = np.argmax(downsampled_correlation_clip)
        # print("very_short_clip downsampled_correlation_clip_peak", downsampled_correlation_clip_peak)
        area_of_overlap = area_of_overlap_ratio(clip_within_peak,
                                                correlation_slice_within_peak)
        return area_of_overlap,{"clip_within_peak":clip_within_peak,"correlation_slice_within_peak":correlation_slice_within_peak}

    #     # repeating = False faster than repeat method and picks up those soft ones
    #     # won't work well if there are multiple occurrences of the same clip
    #     # because it only picks the loudest one or the most matching one
    # won't work well for very short clips like single beep
    # because it is more likely to have false positives
    def _correlation_method(self, clip_data, audio_section, sr, index, seconds_per_chunk,repeating=True):
        clip, clip_name, sliding_window, correlation_clip = (
            itemgetter("clip","clip_name","sliding_window","correlation_clip")(clip_data))
        debug_mode = self.debug_mode

        clip_length = len(clip)

        very_short_clip = len(clip) < 0.75 * sr

        if repeating: #eliminate obviously bad ones by comparing with original clip and select by certain threshold
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

        if repeating:
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
            # slice
            correlation_slice = slicing_with_zero_padding(correlation, len(correlation_clip), peak)
            correlation_slice = correlation_slice/np.max(correlation_slice)

            #if np.argmax(correlation_slice) != np.argmax(correlation_clip):
            #    raise ValueError(f"peak {np.argmax(correlation_slice)} not aligned with the original clip {np.argmax(correlation_clip)}, potential bug in the middle of the chain")

            similarity = self._calculate_similarity(correlation_slice=correlation_slice, correlation_clip=correlation_clip)

            if debug_mode:
                print("similarity", similarity)
                #if similarity <= 0.01:
                similarities.append(similarity)
                correlation_slices.append(correlation_slice)

            if very_short_clip:
                if similarity > self.very_short_clip_similarity_threshold:
                    if debug_mode:
                        print(f"failed verification for {section_ts} due to similarity {similarity} > {self.very_short_clip_similarity_threshold}")
                elif self.similarity_method == "mse" and similarity > self.very_short_clip_similarity_threshold_conditional:
                    area_of_overlap,_ = self._calculate_area_of_overlap_ratio(correlation_clip,
                                                                            correlation_slice)

                    if area_of_overlap < 0.07:
                        peaks_final.append(peak)
                    else:
                        if debug_mode:
                            print(f"failed verification for very short clip {section_ts} due to area_of_overlap {area_of_overlap} > 0.07")
                else:
                    peaks_final.append(peak)
            else:
                if similarity > self.similarity_threshold:
                    if debug_mode:
                        print(f"failed verification for {section_ts} due to similarity {similarity} > {self.similarity_threshold}")
                else:
                    peaks_final.append(peak)

        if debug_mode:
            filtered_similarity = []
            for i,peak in enumerate(peaks):
                similarity = similarities[i]
                correlation_slice = correlation_slices[i]

                if self.similarity_method == "mse":
                    graph_max = 0.01
                elif self.similarity_method == "mae":
                    graph_max = 0.05
                else:
                    raise ValueError("unknown similarity method")
                if graph_max is None or similarity <= graph_max:
                    filtered_similarity.append(similarity)
                    graph_dir = f"./tmp/graph/cross_correlation_slice/{clip_name}"
                    os.makedirs(graph_dir, exist_ok=True)

                    # Optional: plot the correlation graph to visualize
                    plt.figure(figsize=(10, 4))
                    plt.plot(correlation_slice)
                    plt.title('Cross-correlation between the audio clip and full track before slicing')
                    plt.xlabel('Lag')
                    plt.ylabel('Correlation coefficient')
                    plt.savefig(
                        f'{graph_dir}/{clip_name}_{index}_{section_ts}_{peak}.png')
                    plt.close()

            if len(filtered_similarity) > 0:
                peak_dir = f"./tmp/debug/cross_correlation_{clip_name}"
                os.makedirs(peak_dir, exist_ok=True)
                seconds=[]
                areas = []
                #distances = []

                peak_profiles=[]
                for i,item in enumerate(peaks):
                    seconds.append(item / sr)
                    correlation_slice = correlation_slices[i]
                    # #area_of_overlap = area_of_overlap_ratio(correlation_clip, correlation_slice)


                    area_of_overlap,props = self._calculate_area_of_overlap_ratio(correlation_clip,
                                                                            correlation_slice)

                    clip_within_peak = props["clip_within_peak"]
                    correlation_slice_within_peak = props["correlation_slice_within_peak"]

                    graph_dir = f"./tmp/graph/cross_correlation_slice_downsampled/{clip_name}"
                    os.makedirs(graph_dir, exist_ok=True)

                    plt.figure(figsize=(10, 4))
                    plt.plot(correlation_slice_within_peak)
                    plt.plot(clip_within_peak)
                    plt.title('Cross-correlation between the audio clip and full track before slicing')
                    plt.xlabel('Lag')
                    plt.ylabel('Correlation coefficient')
                    plt.savefig(
                        f'{graph_dir}/{clip_name}_{index}_{section_ts}_{item}.png')
                    plt.close()

                    areas.append(area_of_overlap)

                print(json.dumps({"peaks":peaks,"seconds":seconds,
                                  "areas":areas,
                                  #"pdc":pdc,
                                  #"peak_profiles":peak_profiles,
                                  #"pds":pds,
                                  #"properties":properties,
                                  "similarities":similarities}, indent=2,cls=NumpyEncoder), file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))
            self.similarity_debug[clip_name].append(filtered_similarity)

            print(f"---")

        peak_times = np.array(peaks_final) / sr

        return peak_times


