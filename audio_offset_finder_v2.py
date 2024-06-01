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

from dtaidistance import dtw
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

from numpy_encoder import NumpyEncoder
from peak_methods import get_peak_profile
from utils import slicing_with_zero_padding, downsample

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

def dtw_distance(series1, series2):
    d = dtw.distance(series1, series2)
    return d

class AudioOffsetFinder:
    def __init__(self, clip_paths, method=DEFAULT_METHOD,debug_mode=False):
        self.clip_paths = clip_paths
        self.method = method
        self.debug_mode = debug_mode
        #self.correlation_cache_correlation_method = {}
        self.normalize = True
        self.target_sample_rate = 8000
        self.similarity_debug=defaultdict(list)
        self.similarity_debug_repeat=defaultdict(list)
        self.similarity_method = "mse"
        if self.similarity_method == "mse":
            self.similarity_threshold = 0.005
        elif self.similarity_method == "mae":
            self.similarity_threshold = 0.05
        else:
            raise ValueError("unknown similarity method")
        self.downsample = False

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
                print("clip_length", len(clip))
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

            if self.method == "correlation":
                downsampling_factor = 1
                # need this for some clips like single beep
                if self.downsample:
                    downsampling_factor = int(len(correlation_clip) / 200)
            else: # does not support downsampling yet
                downsampling_factor = 1

            if downsampling_factor > 1:
                downsampled_correlation_clip = downsample(correlation_clip, downsampling_factor)

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
            else:
                downsampled_correlation_clip = None
            #downsampled_correlation_clip = downsample(correlation_clip, int(len(correlation_clip) / 100))

            clip_datas[clip_path] = {"clip":clip,
                                     "clip_name":clip_name,
                                     "sliding_window":sliding_window,
                                     "correlation_clip":correlation_clip,
                                     "downsampling_factor":downsampling_factor,
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

        if self.debug_mode and self.method == "non_repeating_correlation":
            for clip_path in clip_paths:
                clip_name, _ = os.path.splitext(os.path.basename(clip_path))
                #print("self.similarity_debug[clip_name]",self.similarity_debug[clip_name])
                graph_dir = f"./tmp/graph/{self.method}_similarity_{self.similarity_method}"
                os.makedirs(graph_dir, exist_ok=True)

                # Optional: plot the correlation graph to visualize
                plt.figure(figsize=(10, 4))
                plt.plot(self.similarity_debug[clip_name])
                plt.title('similarity')
                plt.xlabel('index')
                plt.ylabel('value')
                plt.savefig(
                    f'{graph_dir}/{clip_name}.png')
                plt.close()

        if self.debug_mode and self.method == "correlation":
            for clip_path in clip_paths:
                clip_name, _ = os.path.splitext(os.path.basename(clip_path))
                #print("self.similarity_debug[clip_name]",self.similarity_debug[clip_name])
                graph_dir = f"./tmp/graph/{self.method}_similarity_{self.similarity_method}"
                os.makedirs(graph_dir, exist_ok=True)

                x_coords = []
                y_coords = []

                for index,arr in enumerate(self.similarity_debug_repeat[clip_name]):
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
                                            seconds_per_chunk=seconds_per_chunk,

                                                  )
        elif self.method == "non_repeating_correlation":
            peak_times = self._non_repeating_correlation(clip_data, audio_section=audio_section, sr=sr, index=index,
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

    def _calculate_similarity(self, correlation_clip, correlation_slice):

        if self.similarity_method == "mae":
            similarity = mean_absolute_error(correlation_clip, correlation_slice)
        elif self.similarity_method == "mse":
            similarity = mean_squared_error(correlation_clip, correlation_slice)
        else:
            raise ValueError("unknown similarity method")
        return similarity

    # won't work well for very short clips like single beep
    # because it is more likely to have false positives
    def _correlation_method(self, clip_data, audio_section, sr, index, seconds_per_chunk):
        clip, clip_name, sliding_window, correlation_clip, downsampling_factor, downsampled_correlation_clip = (
            itemgetter("clip","clip_name","sliding_window","correlation_clip", "downsampling_factor","downsampled_correlation_clip")(clip_data))
        debug_mode = self.debug_mode
        threshold = 0.25

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

        # if debug_mode:
        #     percentile = np.percentile(correlation, 99)
        #     max_correlation = max(correlation)
        #     ratio = percentile / max_correlation
        #     print(f"{section_ts} percentile", percentile)
        #     print(f"{section_ts} max correlation: {max_correlation}")
        #     print(f"{section_ts} ratio: {ratio}")
        #     print(f"---")

        #height = threshold
        distance = clip_length
        #width = int(max(clip_length, 1 * sr) / 512)
        # find the peaks in the spectrogram
        #peaks, properties = find_peaks(correlation, prominence=threshold, width=[0,width], distance=distance)
        peaks, properties = find_peaks(correlation, height=threshold, distance=distance)


        peaks_final = []

        # for debugging
        similarities = []
        correlation_slices = []

        for peak in peaks:
            # slice
            correlation_slice = slicing_with_zero_padding(correlation, len(correlation_clip), peak)
            correlation_slice = correlation_slice/np.max(correlation_slice)

            clip_compare = correlation_clip

            if downsampling_factor > 1:
                correlation_slice = downsample(correlation_slice, downsampling_factor)
                clip_compare = downsampled_correlation_clip

            similarity = self._calculate_similarity(correlation_slice=correlation_slice, correlation_clip=clip_compare)

            if debug_mode:
                print("similarity", similarity)
                #if similarity <= 0.01:
                similarities.append(similarity)
                correlation_slices.append(correlation_slice)

            if similarity > self.similarity_threshold:
                if debug_mode:
                    print(f"failed verification for {section_ts} due to similarity {similarity} > {self.similarity_threshold}")
            else:
                peaks_final.append(peak)

        if debug_mode:
            filtered_similarity = []
            for ipeak,peak in enumerate(peaks):
                similarity = similarities[ipeak]
                correlation_slice = correlation_slices[ipeak]
                threshold = 0.01 if self.similarity_method == "mse" else 0.1
                if similarity <= threshold:
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
                # peak_profiles=[]
                # for i,item in enumerate(peaks):
                #     correlation_slice = correlation_slices[i]
                #     peak_profile_slice = get_peak_profile(np.argmax(correlation_slice), correlation_slice)
                #     peak_profiles.append(peak_profile_slice)
                #     seconds.append(item/sr)
                # #peaks_test.append()
                print(json.dumps({"peaks":peaks,"seconds":seconds,"properties":properties,"similarities":similarities}, indent=2,cls=NumpyEncoder), file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))
            self.similarity_debug_repeat[clip_name].append(filtered_similarity)

        if debug_mode:
            print(f"---")

        peak_times = np.array(peaks_final) / sr

        return peak_times

    # faster than repeat method and picks up those soft ones
    # won't work well if there are multiple occurrences of the same clip
    # because it only picks the loudest one or the most matching one
    def _non_repeating_correlation(self, clip_data, audio_section, sr, index, seconds_per_chunk):
        clip, clip_name, sliding_window, correlation_clip = itemgetter("clip",
                                                                     "clip_name",
                                                                     "sliding_window",
                                                                     "correlation_clip",
                                                                     )(clip_data)
        debug_mode = self.debug_mode
        section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)

        # Cross-correlate and normalize correlation
        correlation = correlate(audio_section, clip, mode='full', method='fft')

        # abs
        correlation = np.abs(correlation)
        correlation /= np.max(correlation)

        max_index = np.argmax(correlation)

        if debug_mode:
            # audio section
            print("audio_section length", len(audio_section))

            print("correlation length", len(correlation))
            graph_dir = f"./tmp/graph/non_repeating_cross_correlation/{clip_name}"
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


        # slice
        correlation_slice = slicing_with_zero_padding(correlation, len(correlation_clip), max_index)

        similarity = self._calculate_similarity(correlation_slice=correlation_slice, correlation_clip=correlation_clip)

        if debug_mode:
            graph_dir = f"./tmp/graph/non_repeat_cross_correlation_slice/{clip_name}"
            os.makedirs(graph_dir, exist_ok=True)

            # Optional: plot the correlation graph to visualize
            plt.figure(figsize=(10, 4))
            plt.plot(correlation_slice)
            plt.title('Cross-correlation between the audio clip and full track before slicing')
            plt.xlabel('Lag')
            plt.ylabel('Correlation coefficient')
            plt.savefig(
                f'{graph_dir}/{clip_name}_{index}_{section_ts}.png')
            plt.close()

            self.similarity_debug[clip_name].append(similarity)

            print(f"{section_ts} similarity",similarity)

            debug_dir = f"./tmp/debug/non_repeating_cross_correlation_{clip_name}"
            os.makedirs(debug_dir, exist_ok=True)
            print(json.dumps({"max_index":max_index,
                              "similarity":similarity,
                              }, indent=2, cls=NumpyEncoder),
                  file=open(f'{debug_dir}/{clip_name}_{index}_{section_ts}.txt', 'w'))

        if similarity > self.similarity_threshold:
            if debug_mode:
                print(f"failed verification for {section_ts} due to similarity {similarity} > {self.similarity_threshold}")
            return []
        else:
            return [max_index / sr]
