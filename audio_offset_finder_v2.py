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
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error

from numpy_encoder import NumpyEncoder
from peak_methods import get_peak_profile, find_closest_troughs
from utils import slicing_with_zero_padding, downsample, area_of_overlap_ratio

logger = logging.getLogger(__name__)

#ignore possible clipping
warnings.filterwarnings('ignore', module='pyloudnorm')

DEFAULT_METHOD="correlation"

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


# def dtw_distance(series1, series2):
#     distance, path = fastdtw(series1, series2, dist=2)
#     return distance
#     #return d


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
    SIMILARITY_METHOD_MEAN_SQUARED_ERROR = "mean_squared_error"
    SIMILARITY_METHOD_MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    SIMILARITY_METHOD_MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    #SIMILARITY_METHOD_TEST = "test"
    def __init__(self, clip_paths, method=DEFAULT_METHOD,debug_mode=False):
        self.clip_paths = clip_paths
        self.method = method
        self.debug_mode = debug_mode
        #self.correlation_cache_correlation_method = {}
        self.normalize = True
        self.target_sample_rate = 8000
        self.target_num_sample_after_resample = 1001
        self.similarity_debug=defaultdict(list)
        self.max_distance_debug=defaultdict(list)
        #self.areas_debug=defaultdict(list)
        self.similarity_method = self.SIMILARITY_METHOD_MEAN_SQUARED_ERROR
        match self.similarity_method:
            case self.SIMILARITY_METHOD_MEAN_SQUARED_ERROR:
                self.similarity_threshold = 0.002
            case self.SIMILARITY_METHOD_MEAN_ABSOLUTE_ERROR:
                self.similarity_threshold = 0.02
            case self.SIMILARITY_METHOD_MEDIAN_ABSOLUTE_ERROR: #median_absolute_error, a bit better for news report beep
                self.similarity_threshold = 0.02
            # case self.SIMILARITY_METHOD_TEST:
            #     #test
            #     self.similarity_threshold = 0.02
            case _:
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

            sliding_window = self._get_chunking_timing_info(clip_name,clip_seconds,seconds_per_chunk)

            if self.normalize:
                # max_loudness = np.max(np.abs(clip))
                # clip = clip / max_loudness
                sr = self.target_sample_rate
                #clip_second = clip_length / sr

                # normalize loudness
                if clip_seconds < 0.5:
                    meter = pyln.Meter(sr, block_size=clip_seconds)
                else:
                    meter = pyln.Meter(sr)  # create BS.1770 meter
                loudness = meter.integrated_loudness(clip)

                # loudness normalize audio to -16 dB LUFS
                clip = pyln.normalize.loudness(clip, loudness, -16.0)

                # if self.debug_mode:
                #     audio_test_dir = f"./tmp/clip_audio_normalized"
                #     os.makedirs(audio_test_dir, exist_ok=True)
                #     sf.write(f"{audio_test_dir}/{clip_name}.wav", clip, self.target_sample_rate)

            correlation_clip,absolute_max = self._get_clip_correlation(clip, clip_name)

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

            is_news_report_beep = clip_name == "rthk_beep"
            if is_news_report_beep:
                downsampled_correlation_clip = downsample_preserve_maxima(correlation_clip, self.target_num_sample_after_resample)
            else:
                downsampled_correlation_clip = None
            #downsampled_correlation_clip = downsample(correlation_clip, len(correlation_clip)//500)
            #print(f"downsampled_correlation_clip {clip_name} length", len(downsampled_correlation_clip))
            #exit(1)

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
                                     "correlation_clip":correlation_clip,
                                     "correlation_clip_absolute_max":absolute_max,
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

        if self.debug_mode and self.method == "correlation":
            for clip_path in clip_paths:
                clip_name, _ = os.path.splitext(os.path.basename(clip_path))

                # similarity debug
                graph_dir = f"./tmp/graph/{self.method}_similarity_{self.similarity_method}/{clip_name}"
                os.makedirs(graph_dir, exist_ok=True)

                x_coords = []
                y_coords = []

                for index,similarity in self.similarity_debug[clip_name]:
                    x_coords.append(index)
                    y_coords.append(similarity)

                plt.figure(figsize=(10, 4))
                # Create scatter plot
                plt.scatter(x_coords, y_coords)

                if self.similarity_method == self.SIMILARITY_METHOD_MEAN_SQUARED_ERROR and len(y_coords) > 0 and np.max(y_coords) > 0.005:
                    ylimit = max(0.01, np.median(y_coords))
                    # Set the y limits
                    plt.ylim(0, ylimit)
                elif len(y_coords) > 10:
                    ylimit = np.median(y_coords)
                    # Set the y limits
                    plt.ylim(0, ylimit)


                # Adding titles and labels
                plt.title('Scatter Plot for Similarity')
                plt.xlabel('Value')
                plt.ylabel('Sublist Index')
                plt.savefig(
                    f'{graph_dir}/{suffix}.png')
                plt.close()

        process.wait()

        return all_peak_times

    def _get_chunking_timing_info(self, clip_name, clip_seconds, seconds_per_chunk):
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

    def _get_clip_correlation(self, clip, clip_name):
        # Cross-correlate and normalize correlation
        correlation_clip = correlate(clip, clip, mode='full', method='fft')

        # abs
        correlation_clip = np.abs(correlation_clip)
        absolute_max = np.max(correlation_clip)
        correlation_clip /= absolute_max

        return correlation_clip,absolute_max


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
            #max_loudness = np.max(np.abs(audio_section))
            #audio_section = audio_section / max_loudness
            audio_section_seconds = len(audio_section) / sr
            #normalize loudness
            if audio_section_seconds < 0.5:
                # not sure if there are valid use cases for this
                #raise ValueError("audio_section_seconds < 0.5 second")
                meter = pyln.Meter(sr, block_size=audio_section_seconds)
            else:
                meter = pyln.Meter(sr)  # create BS.1770 meter

            loudness = meter.integrated_loudness(audio_section)

            # loudness normalize audio to -16 dB LUFS
            audio_section = pyln.normalize.loudness(audio_section, loudness, -16.0)
            # if self.debug_mode:
            #     section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)
            #     audio_test_dir = f"./tmp/audio_section_normalized"
            #     os.makedirs(audio_test_dir, exist_ok=True)
            #     sf.write(f"{audio_test_dir}/{clip_name}_{index}_{section_ts}.wav", audio_section, self.target_sample_rate)


        # if debug_mode:
        #     os.makedirs("./tmp/audio", exist_ok=True)
        #     sf.write(
        #         f"./tmp/audio/section_{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.wav",
        #         audio_section, sr)

        if self.method == "correlation":

            # samples_skip_end does not skip results from being included yet
            peak_times = self._correlation_method(clip_data, audio_section=audio_section, sr=sr, index=index,
                                                  seconds_per_chunk=seconds_per_chunk,
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

    def _get_max_distance(self, downsampled_correlation_clip, downsampled_correlation_slice,):
        distances = np.abs(downsampled_correlation_clip-downsampled_correlation_slice)
        max_distance_index = np.argmax(distances)
        max_distance = distances[max_distance_index]

        return max_distance,max_distance_index


    # won't work well for very short clips like single beep
    # because it is more likely to have false positives or miss good ones
    def _correlation_method(self, clip_data, audio_section, sr, index, seconds_per_chunk):
        clip, clip_name, sliding_window, correlation_clip, correlation_clip_absolute_max, downsampled_correlation_clip= (
            itemgetter("clip","clip_name","sliding_window","correlation_clip","correlation_clip_absolute_max","downsampled_correlation_clip")(clip_data))

        is_news_report_beep = clip_name == "rthk_beep"

        debug_mode = self.debug_mode

        clip_length = len(clip)

        #very_short_clip = len(clip) < 0.75 * sr

        # zeroes_second_pad = 1
        # # pad zeros between audio and clip
        #zeroes = np.zeros(clip_length + zeroes_second_pad * sr)
        #audio = np.concatenate((audio_section, zeroes))
        # samples_skip_end = zeroes_second_pad * sr + clip_length

        audio=audio_section

        # Cross-correlate and normalize correlation
        correlation = correlate(audio, clip, mode='full', method='fft')
        # abs
        correlation = np.abs(correlation)
        absolute_max = np.max(correlation)
        max_choose = max(correlation_clip_absolute_max,absolute_max)
        correlation /= max_choose

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

        #max_sample = len(audio) - samples_skip_end
        #trim placeholder clip
        #correlation = correlation[:max_sample]


        distance = clip_length
        height_min = 0.25
        peaks, _ = find_peaks(correlation, height=height_min, distance=distance)

        peaks_final = []

        # for debugging
        similarities = []
        peaks_debug=[]
        correlation_slices = []
        seconds = []

        for peak in peaks:
            after = peak + len(correlation_clip)//2
            before = peak - len(correlation_clip)//2
            # max_height_index = np.argmax(correlation)
            # if one_shot and peak != max_height_index:
            #     logger.warning(f"peak {peak} not equal to max_height_index {max_height_index}")
            #print("peak after",after)
            #print("len(correlation)",len(correlation))
            if after > len(correlation)-1+2:
                logger.warning(f"peak {peak} after is {after} > len(correlation)+2 {len(correlation)+2}, skipping")
                continue
            elif before < -2:
                logger.warning(f"peak {peak} before is {before} < -2, skipping")
                continue

            # slice
            correlation_slice = slicing_with_zero_padding(correlation, len(correlation_clip), peak)
            correlation_slice = correlation_slice/np.max(correlation_slice)

            if len(correlation_slice) != len(correlation_clip):
                raise ValueError(f"correlation_slice length {len(correlation_slice)} not equal to correlation_clip length {len(correlation_clip)}")

            # downsample
            if is_news_report_beep:
                #correlation_clip = downsampled_correlation_clip
                downsampled_correlation_slice = downsample_preserve_maxima(correlation_slice, self.target_num_sample_after_resample)
                if self.similarity_method != self.SIMILARITY_METHOD_MEAN_SQUARED_ERROR:
                    raise ValueError("only mean_squared_error is supported for news report beep")
                similarity = mean_squared_error(downsampled_correlation_clip, downsampled_correlation_slice)
                #similarity_middle = np.mean(similarity_partitions[4:6])
                similarity_whole = similarity
                similarity_left = 0
                similarity_middle = 0
                similarity_right = 0
            else:
                partition_count = 10
                partition_size = len(correlation_clip) // partition_count
                #quarter = len(downsampled_correlation_clip) // 4

                match self.similarity_method:
                    case self.SIMILARITY_METHOD_MEAN_SQUARED_ERROR:
                        similarity_partitions=[]
                        for i in range(partition_count):
                            similarity_partitions.append(mean_squared_error(correlation_clip[i*partition_size:(i+1)*partition_size],
                                                                           correlation_slice[i*partition_size:(i+1)*partition_size]))

                        # similarity_left = (similarity_quadrants[0]+similarity_quadrants[1])/2
                        # similarity_middle = (similarity_quadrants[1]+similarity_quadrants[2])/2
                        # similarity_right = (similarity_quadrants[2]+similarity_quadrants[3])/2
                        # similarity_whole = (similarity_left + similarity_right) / 2
                        # # clip the fat tails
                        # if similarity_middle < similarity_whole:
                        #     similarity = similarity_middle
                        # else:
                        #     similarity = similarity_whole

                        # real distortions happen in the middle most of the time except for news report beep
                        similarity_middle = np.mean(similarity_partitions[4:6])
                        similarity_whole = np.mean(similarity_partitions)
                        similarity_left = 0
                        #similarity_middle = 0
                        similarity_right = 0

                        #similarity = similarity_middle
                        similarity = min(similarity_whole,similarity_middle)

                        #similarity = min(similarity_left,similarity_middle,similarity_right)
                        #similarity = similarity_whole = (similarity_left + similarity_right)/2
                    case self.SIMILARITY_METHOD_MEAN_ABSOLUTE_ERROR:
                        raise NotImplementedError("mean_absolute_error not implemented")
                        # similarity_quadrants = []
                        # for i in range(4):
                        #     similarity_quadrants.append(mean_absolute_error(correlation_clip[i*quarter:(i+1)*quarter],correlation_slice[i*quarter:(i+1)*quarter]))
                        #
                        # similarity_left = (similarity_quadrants[0]+similarity_quadrants[1])/2
                        # similarity_middle = (similarity_quadrants[1]+similarity_quadrants[2])/2
                        # similarity_right = (similarity_quadrants[2]+similarity_quadrants[3])/2
                        # similarity = similarity_whole = (similarity_left + similarity_right) / 2
                        #similarity = min(similarity_left,similarity_middle,similarity_right)
                    case self.SIMILARITY_METHOD_MEDIAN_ABSOLUTE_ERROR:
                        similarity = median_absolute_error(correlation_clip,correlation_slice)
                        similarity_whole = similarity
                        similarity_left = 0
                        similarity_middle = 0
                        similarity_right = 0
                    case _:
                        raise ValueError("unknown similarity method")

            if debug_mode:
                print("similarity", similarity)
                seconds.append(peak / sr)
                self.similarity_debug[clip_name].append((index, similarity,))

                if is_news_report_beep:
                    correlation_slice_graph = downsampled_correlation_slice
                    correlation_clip_graph = downsampled_correlation_clip
                else:
                    correlation_slice_graph = correlation_slice
                    correlation_clip_graph = correlation_clip

                graph_max = 0.01
                if similarity <= graph_max:
                    graph_dir = f"./tmp/graph/cross_correlation_slice/{clip_name}"
                    os.makedirs(graph_dir, exist_ok=True)

                    # Optional: plot the correlation graph to visualize
                    plt.figure(figsize=(10, 4))
                    plt.plot(correlation_slice_graph)
                    plt.plot(correlation_clip_graph, alpha=0.7)
                    plt.title('Cross-correlation between the audio clip and full track before slicing')
                    plt.xlabel('Lag')
                    plt.ylabel('Correlation coefficient')
                    plt.savefig(
                        f'{graph_dir}/{clip_name}_{index}_{section_ts}_{peak}.png')
                    plt.close()

                similarities.append((similarity,{"whole":similarity_whole,
                                                 "left":similarity_left,
                                                 "middle":similarity_middle,
                                                 "right":similarity_right,
                                                 "left_right_diff": abs(similarity_left-similarity_right),
                                                 }))


            if similarity <= self.similarity_threshold:
                peaks_final.append(peak)
            else:
                if debug_mode:
                    print(f"failed verification for {section_ts} due to similarity {similarity} > {self.similarity_threshold}")

        if debug_mode:
            peak_dir = f"./tmp/debug/cross_correlation_{clip_name}"
            os.makedirs(peak_dir, exist_ok=True)

            print(json.dumps({"peaks": peaks, "seconds": seconds,
                              # "area_props":area_props,
                              "similarities": similarities}, indent=2, cls=NumpyEncoder),
                  file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))

            print(f"---")

        peak_times = np.array(peaks_final) / sr

        return peak_times
