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
from scipy.integrate import simpson
#from pure_tone import is_pure_tone

import soundfile as sf

import numpy as np

from numpy._typing import DTypeLike
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

import ffmpeg

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
from pure_tone import is_news_report_beep, detect_sine_tone
from utils import slicing_with_zero_padding, area_of_overlap_ratio

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

# from librosa.util.buf_to_float
def buf_to_float(
    x: np.ndarray, *, n_bytes: int = 2, dtype: DTypeLike = np.float32
) -> np.ndarray:
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer
    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``
    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """
    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = f"<i{n_bytes:d}"

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)

def convert_audio_arr_to_float(audio):
    #raise "chafa"
    return buf_to_float(audio,n_bytes=2, dtype='float32')


# def dtw_distance(series1, series2):
#     distance, path = fastdtw(series1, series2, dist=2)
#     return distance
#     #return d


# # Normalize the curves to the same scale
# def normalize_curve(curve):
#     return (curve - np.min(curve)) / (np.max(curve) - np.min(curve))

# # Apply DTW and warp the target curve
# def warp_with_dtw(reference, target):
#     # Compute dynamic time warping path
#     path = dtaidistance.dtw.warping_path(reference, target)
#     #print("path",path)
#
#     # Create an array to hold the warped target
#     warped_target = np.zeros_like(reference)
#     for ref_idx, target_idx in path:
#         warped_target[ref_idx] = target[target_idx]
#
#     return warped_target, path

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
    # SIMILARITY_METHOD_MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    # SIMILARITY_METHOD_MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    #SIMILARITY_METHOD_TEST = "test"

    # won't work well for short clips because false positives get low similarity
    clip_properties = {
        # "受之有道outro": {
        #     # triangular shape at the bottom occupying large area
        #     "mean_squared_error_similarity_threshold": 0.005,
        # },
        # "temple_bell": {
        #     # triangular shape at the bottom occupying large area
        #     "mean_squared_error_similarity_threshold": 0.01,
        # },
        "rthk_beep": {
            # short pure tone needs quite a bit of workaround
            # need to downsample and no partition and check cross similarity only
            # won't partition or calculate area ratio if downsample
            "is_pure_tone_pattern": True,
        },
    }
    def __init__(self, clip_paths, method=DEFAULT_METHOD,debug_mode=False):
        self.clip_paths = clip_paths
        self.method = method
        self.debug_mode = debug_mode
        #self.correlation_cache_correlation_method = {}
        self.normalize = True
        self.target_sample_rate = 8000
        self.similarity_method = self.SIMILARITY_METHOD_MEAN_SQUARED_ERROR
        #match self.similarity_method:
        #    case self.SIMILARITY_METHOD_MEAN_SQUARED_ERROR:
        #self.similarity_threshold = 0.01
            # case self.SIMILARITY_METHOD_MEAN_ABSOLUTE_ERROR:
            #     self.similarity_threshold = 0.02
            # case self.SIMILARITY_METHOD_MEDIAN_ABSOLUTE_ERROR: #median_absolute_error, a bit better for news report beep
            #     self.similarity_threshold = 0.02
            # case self.SIMILARITY_METHOD_TEST:
            #     #test
            #     self.similarity_threshold = 0.02
            # case _:
            #     raise ValueError("unknown similarity method")

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
        clip_cache={
            "downsampled_correlation_clips":{},
            "similarity_debug":defaultdict(list),
        }

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

            #is_pure_tone_pattern = self.clip_properties.get(clip_name, {}).get("is_pure_tone_pattern", False)
            #downsampled_correlation_clip = None
            # if is_pure_tone_pattern:
            #     downsampled_correlation_clip = downsample_preserve_maxima(correlation_clip, self.beep_target_num_sample_after_resample)
            #
            #     #downsampled_correlation_clip = downsample(correlation_clip, len(correlation_clip)//500)
            #     #print(f"downsampled_correlation_clip {clip_name} length", len(downsampled_correlation_clip))
            #     #exit(1)
            #
            #     if self.debug_mode:
            #
            #         print("average correlation_clip", np.mean(correlation_clip))
            #         print("average downsampled_correlation_clip", np.mean(downsampled_correlation_clip))
            #
            #         #self.debug_clip_area(correlation_clip)
            #
            #         graph_dir = f"./tmp/graph/clip_correlation_downsampled"
            #         os.makedirs(graph_dir, exist_ok=True)
            #
            #         plt.figure(figsize=(10, 4))
            #
            #         plt.plot(downsampled_correlation_clip)
            #         plt.title('Cross-correlation of the audio clip itself')
            #         plt.xlabel('Lag')
            #         plt.ylabel('Correlation coefficient')
            #         plt.savefig(
            #             f'{graph_dir}/{clip_name}.png')
            #         plt.close()

            clip_datas[clip_path] = {"clip":clip,
                                     "clip_name":clip_name,
                                     "sliding_window":sliding_window,
                                     "correlation_clip":correlation_clip,
                                     "correlation_clip_absolute_max":absolute_max,
                                     #"downsampled_correlation_clip":downsampled_correlation_clip,
                                     }

        #exit(1)

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
                                                 clip_cache=clip_cache,
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

                for index,similarity in clip_cache['similarity_debug'][clip_name]:
                    x_coords.append(index)
                    y_coords.append(similarity)

                plt.figure(figsize=(10, 4))
                # Create scatter plot
                plt.scatter(x_coords, y_coords)


                ylimit = max(0.01, np.median(y_coords))
                # Set the y limits
                plt.ylim(0, ylimit)

                # Adding titles and labels
                plt.title('Scatter Plot for Similarity')
                plt.xlabel('Value')
                plt.ylabel('Sublist Index')
                plt.savefig(
                    f'{graph_dir}/{suffix}.png')
                plt.close()

                # # distance debug
                # graph_dir = f"./tmp/graph/{self.method}_distance_{self.similarity_method}/{clip_name}"
                # os.makedirs(graph_dir, exist_ok=True)
                #
                # x_coords = []
                # y_coords = []
                #
                # for index,distance,distance_index in self.max_distance_debug[clip_name]:
                #     x_coords.append(index)
                #     y_coords.append(distance)
                #
                # plt.figure(figsize=(10, 4))
                # # Create scatter plot
                # plt.scatter(x_coords, y_coords)
                #
                # # Adding titles and labels
                # plt.title('Scatter Plot for Distance')
                # plt.xlabel('Value')
                # plt.ylabel('Sublist Index')
                # plt.savefig(
                #     f'{graph_dir}/{suffix}.png')
                # plt.close()

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
    def _process_chunk(self, chunk, clip_data, clip_cache, sr, previous_chunk, index, seconds_per_chunk):
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
                                                  clip_cache=clip_cache,
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

    def _calculate_area_of_overlap_ratio(self, correlation_clip, correlation_slice):

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

        # # static
        # middle = len(downsampled_correlation_clip) // 2
        #
        # # mid point of the peak for 1/10th of the downsampled_correlation_clip
        # new_left = middle - len(downsampled_correlation_clip) // 20
        # new_right = middle + len(downsampled_correlation_clip) // 20
        #
        # clip_within_peak = downsampled_correlation_clip[new_left:new_right]
        # correlation_slice_within_peak = downsampled_correlation_slice[new_left:new_right]

        area_of_overlap, props = area_of_overlap_ratio(correlation_clip,
                                                       correlation_slice)

        return area_of_overlap,{
            #"clip_within_peak":clip_within_peak,
            #"correlation_slice_within_peak":correlation_slice_within_peak,
            #"downsampled_correlation_slice":downsampled_correlation_slice,
            #"new_left":new_left,
            #"new_right":new_right,
            "area_props":props,
        }

    def _get_max_distance(self, downsampled_correlation_clip, downsampled_correlation_slice,):
        distances = np.abs(downsampled_correlation_clip-downsampled_correlation_slice)
        max_distance_index = np.argmax(distances)
        max_distance = distances[max_distance_index]

        return max_distance,max_distance_index


    # won't work well for very short clips like single beep
    # because it is more likely to have false positives or miss good ones
    def _correlation_method(self, clip_data, clip_cache, audio_section, sr, index, seconds_per_chunk):
        clip, clip_name, sliding_window, correlation_clip, correlation_clip_absolute_max= (
            itemgetter("clip","clip_name","sliding_window","correlation_clip","correlation_clip_absolute_max")(clip_data))

        clip_properties = self.clip_properties.get(clip_name, {})
        is_pure_tone_pattern = clip_properties.get("is_pure_tone_pattern", False)


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
            print(f"section_ts: {section_ts}, index {index}")
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

        # no repetition within the duration of the clip
        distance = clip_length
        # minimum height of the peak of the correlation, don't set it too high otherwise will miss some
        # the selected ones are going to be checked for similarity
        # before adding to final peaks
        height_min = 0.25
        peaks, _ = find_peaks(correlation, height=height_min, distance=distance)

        peaks_final = []

        # for debugging
        area_props = []
        similarities = []
        seconds = []
        #distances = []

        for peak in peaks:
            after = peak + len(correlation_clip)//2
            before = peak - len(correlation_clip)//2
            # max_height_index = np.argmax(correlation)
            # if one_shot and peak != max_height_index:
            #     logger.warning(f"peak {peak} not equal to max_height_index {max_height_index}")
            #print("peak after",after)
            #print("len(correlation)",len(correlation))
            if after > len(correlation)+5:
                logger.warning(f"{section_ts} {clip_name} peak {peak} after is {after} > len(correlation)+5 {len(correlation)+5}, skipping")
                continue
            elif before < -5:
                logger.warning(f"{section_ts} {clip_name} peak {peak} before is {before} < -5, skipping")
                continue

            # slice
            correlation_slice = slicing_with_zero_padding(correlation, len(correlation_clip), peak)
            correlation_slice = correlation_slice/np.max(correlation_slice)

            if len(correlation_slice) != len(correlation_clip):
                raise ValueError(f"correlation_slice length {len(correlation_slice)} not equal to correlation_clip length {len(correlation_clip)}")

            if is_pure_tone_pattern:
                self._get_peak_times_beep_v2(
                                                 audio=audio[peak - len(clip):peak + len(clip)],
                                                 peak=peak,
                                                 peaks_final=peaks_final,
                                                 clip_cache=clip_cache,
                                                 area_props=area_props,
                                                 clip_name=clip_name,
                                                 index=index,
                                                 section_ts=section_ts,)
            else:
                self._get_peak_times_normal(correlation_clip=correlation_clip,
                                                correlation_slice=correlation_slice,
                                                seconds=seconds,
                                                peak=peak,
                                                clip_name=clip_name,
                                                index=index,
                                                section_ts=section_ts,
                                                similarities=similarities,
                                                peaks_final=peaks_final,
                                                area_props=area_props,
                                                clip_cache=clip_cache)


        if debug_mode and len(peaks) > 0:
            peak_dir = f"./tmp/debug/cross_correlation_{clip_name}"
            os.makedirs(peak_dir, exist_ok=True)

            print(json.dumps({"peaks": peaks, "seconds": seconds,
                              "area_props":area_props,
                              #"distances": distances,
                              "similarities": similarities}, indent=2, cls=NumpyEncoder),
                  file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))

            print(f"---")

        peak_times = np.array(peaks_final) / sr

        return peak_times

    def _get_peak_times_normal(self, correlation_clip, correlation_slice, seconds, peak, clip_name, index,
                             section_ts, similarities, peaks_final, clip_cache, area_props):

        debug_mode = self.debug_mode
        sr = self.target_sample_rate

        partition_count = 10
        left_bound = 4
        right_bound = 6

        partition_size = len(correlation_clip) // partition_count

        similarity_partitions = []
        for i in range(partition_count):
            similarity_partitions.append(
                mean_squared_error(correlation_clip[i * partition_size:(i + 1) * partition_size],
                                   correlation_slice[i * partition_size:(i + 1) * partition_size]))

        # real distortions happen in the middle most of the time except for news report beep
        # since we don't partition for news report beep, we only need to check the middle
        similarity_middle = np.mean(similarity_partitions[left_bound:right_bound])
        # similarity_whole = np.mean(similarity_partitions)
        similarity_whole = 0
        similarity_left = 0
        similarity_right = 0
        # similarity_left = np.mean(similarity_partitions[0:5])
        # similarity_right = np.mean(similarity_partitions[5:10])

        similarity = similarity_middle
        # similarity = min(similarity_whole,similarity_middle)

        # similarity = min(similarity_left,similarity_middle,similarity_right)
        # similarity = similarity_whole = (similarity_left + similarity_right)/2

        lower_limit = round(len(correlation_clip) * left_bound / partition_count)
        upper_limit = round(len(correlation_clip) * right_bound / partition_count)
        area_overlap_ratio, area_prop = self._calculate_area_of_overlap_ratio(correlation_clip[lower_limit:upper_limit],
                                                                              correlation_slice[
                                                                              lower_limit:upper_limit])

        if debug_mode:
            similarity_debug = clip_cache["similarity_debug"]
            print("similarity", similarity)
            seconds.append(peak / sr)
            similarity_debug[clip_name].append((index, similarity,))

            correlation_slice_graph = correlation_slice
            correlation_clip_graph = correlation_clip

            graph_max = 0.1
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

            area_props.append([area_overlap_ratio, area_prop])

            similarities.append((similarity, {"whole": similarity_whole,
                                              "left": similarity_left,
                                              "middle": similarity_middle,
                                              "right": similarity_right,
                                              "left_right_diff": abs(similarity_left - similarity_right),
                                              }))

        similarity_threshold = 0.01
        similarity_threshold_check_area = 0.002

        # reject if similarity is high enough and little area overlap
        area_overlap_ratio_threshold = 0.5

        if similarity_threshold <= similarity_threshold_check_area:
            raise ValueError(
                f"similarity_threshold {similarity_threshold} needs to be larger than similarity_threshold_check_area {similarity_threshold_check_area}")

        if similarity > similarity_threshold:
            if debug_mode:
                print(f"failed verification for {section_ts} due to similarity {similarity} > {similarity_threshold}")
        # if similarity is between similarity_threshold and similarity_threshold_check_area, check shape ratio
        elif similarity > similarity_threshold_check_area and area_overlap_ratio > area_overlap_ratio_threshold:
            if debug_mode:
                print(
                    f"failed verification for {section_ts} due to area_overlap_ratio {area_overlap_ratio} > {area_overlap_ratio_threshold}")
        else:  # if similarity is less than similarity_threshold_check_area, no need to check area ratio
            peaks_final.append(peak)

    def _get_peak_times_beep_v2(self,audio,peak,peaks_final,clip_cache,area_props,clip_name,index,section_ts):

        sr = self.target_sample_rate
        debug_mode = self.debug_mode

        result = detect_sine_tone(audio, sr)
        detected = result['is_news_report_clip']

        if debug_mode:
            print("detected", detected)
            area_props.append({"detect_sine_tone_result": result})
            audio_test_dir = f"./tmp/clip_audio_news_beep"
            os.makedirs(audio_test_dir, exist_ok=True)
            sf.write(f"{audio_test_dir}/{clip_name}_{index}_{section_ts}_{peak}.wav", audio, self.target_sample_rate)

        if detected:
            peaks_final.append(peak)

    def _get_peak_times_beep(self,correlation_clip,correlation_slice,seconds,peak,clip_name,index,section_ts,similarities,peaks_final,clip_cache,area_props):
        # short beep is very sensitive, it is better to miss some than to have false positives
        similarity_threshold = 0.002

        sr = self.target_sample_rate
        debug_mode = self.debug_mode

        beep_target_num_sample_after_resample = 101

        downsampled_correlation_clip = clip_cache["downsampled_correlation_clips"].get(clip_name)

        if downsampled_correlation_clip is None:
            downsampled_correlation_clip = downsample_preserve_maxima(correlation_clip,beep_target_num_sample_after_resample)
            clip_cache["downsampled_correlation_clips"][clip_name] = downsampled_correlation_clip

        downsampled_correlation_slice = downsample_preserve_maxima(correlation_slice,beep_target_num_sample_after_resample)

        correlation_clip = downsampled_correlation_clip
        correlation_slice = downsampled_correlation_slice

        similarity = mean_squared_error(correlation_clip, correlation_slice)

        similarity_whole = similarity

        #area_overlap_ratio, area_prop = self._calculate_area_of_overlap_ratio(correlation_clip,
        #                                                                      correlation_slice)

        if debug_mode:
            print("similarity", similarity)
            seconds.append(peak / sr)
            similarity_debug = clip_cache["similarity_debug"]
            similarity_debug[clip_name].append((index, similarity,))

            correlation_slice_graph = correlation_slice
            correlation_clip_graph = correlation_clip

            graph_max = 0.1
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

            #area_props.append([area_overlap_ratio, area_prop])

            similarities.append((similarity, {"whole": similarity_whole,
                                              "left": 0,
                                              "middle": 0,
                                              "right": 0,
                                              "left_right_diff": 0,
                                              }))

            if similarity > similarity_threshold:
                if debug_mode:
                    print(
                        f"failed verification for {section_ts} due to similarity {similarity} > {similarity_threshold}")
            else:
                peaks_final.append(peak)