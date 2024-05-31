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

from numpy_encoder import NumpyEncoder
from peak_methods import get_peak_profile
from utils import is_unique_and_sorted, calculate_similarity, slicing_with_zero_padding

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

class AudioOffsetFinder:
    def __init__(self, clip_paths, method=DEFAULT_METHOD,debug_mode=False):
        self.clip_paths = clip_paths
        self.method = method
        self.debug_mode = debug_mode
        self.correlation_cache_correlation_method = {}
        self.target_sample_rate = 8000

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

            clip_datas[clip_path] = {"clip":clip,"clip_name":clip_name,"sliding_window":sliding_window}

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
                #clip_seconds = clip_data["clip_seconds"]
                sliding_window = clip_data["sliding_window"]

                peak_times = self._process_chunk(chunk=chunk, clip=clip, sr=self.target_sample_rate,
                                   previous_chunk=previous_chunk,
                                   sliding_window=sliding_window,
                                   index=i,
                                   clip_name=clip_name,
                                   seconds_per_chunk=seconds_per_chunk,
                                   )

                all_peak_times[clip_path].extend(peak_times)

            # Update previous_chunk to current chunk
            previous_chunk = chunk
            i = i + 1

        process.wait()

        return all_peak_times
    def _get_clip_correlation(self, clip, clip_name):
        # Cross-correlate and normalize correlation
        correlation_clip = correlate(clip, clip, mode='full', method='fft')

        # abs
        correlation_clip = np.abs(correlation_clip)
        correlation_clip /= np.max(correlation_clip)

        if self.debug_mode:
            print("clip_length", len(clip))
            # print("correlation_clip", [float(c) for c in correlation_clip])
            # raise "chafa"
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
        return correlation_clip

    def _get_clip_correlation_cached(self, clip, clip_name, correlation_clip_cache):
        if clip_name in correlation_clip_cache:
            return correlation_clip_cache[clip_name]
        else:
            correlation_clip = self._get_clip_correlation(clip, clip_name)
            correlation_clip_cache[clip_name] = correlation_clip
            return correlation_clip



    # sliding_window: for previous_chunk in seconds from end
    # index: for debugging by saving a file for audio_section
    # seconds_per_chunk: default seconds_per_chunk
    def _process_chunk(self, chunk, clip, sr, previous_chunk, sliding_window, index, seconds_per_chunk, clip_name):
        debug_mode = self.debug_mode
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

        if debug_mode:
            os.makedirs("./tmp/audio", exist_ok=True)
            sf.write(
                f"./tmp/audio/section_{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.wav",
                audio_section, sr)

        if self.method == "correlation":
            # samples_skip_end does not skip results from being included yet
            peak_times = self._correlation_method(clip, audio_section=audio_section, sr=sr, index=index,
                                            seconds_per_chunk=seconds_per_chunk,
                                            clip_name=clip_name)
        elif self.method == "non_repeating_correlation":
            peak_times = self._non_repeating_correlation(clip, audio_section=audio_section, sr=sr, index=index,
                                            seconds_per_chunk=seconds_per_chunk, clip_name=clip_name)
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


    # won't work well for very short clips like single beep
    # because it is more likely to have false positives
    def _correlation_method(self, clip, audio_section, sr, index, seconds_per_chunk, clip_name):
        debug_mode = self.debug_mode
        correlation_clip = self._get_clip_correlation_cached(clip,clip_name,self.correlation_cache_correlation_method)
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
        #width = int(max(clip_length, 1 * sr) / 512)
        # find the peaks in the spectrogram
        #peaks, properties = find_peaks(correlation, prominence=threshold, width=[0,width], distance=distance)
        peaks, properties = find_peaks(correlation, height=threshold, distance=distance)

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

        peaks_final = []
        for peak in peaks:
            # slice
            correlation_slice = slicing_with_zero_padding(correlation, len(correlation_clip), peak)
            correlation_slice = correlation_slice/np.max(correlation_slice)

            if debug_mode:
                graph_dir = f"./tmp/graph/cross_correlation/slice/{clip_name}"
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

            similarity = calculate_similarity(correlation_clip,correlation_slice)

            similarity_threshold = 0.002

            if similarity > similarity_threshold:
                if debug_mode:
                    print(f"failed verification for {section_ts} due to similarity {similarity} > {similarity_threshold}")
                #return []
            else:
                peaks_final.append(peak)
                #return [max_index / sr]

        peak_times = np.array(peaks_final) / sr

        return peak_times

    # faster than repeat method and picks up those soft ones
    # won't work well if there are multiple occurrences of the same clip
    # because it only picks the loudest one or the most matching one
    def _non_repeating_correlation(self, clip, audio_section, sr, index, seconds_per_chunk, clip_name):
        debug_mode = self.debug_mode
        correlation_clip = self._get_clip_correlation_cached(clip, clip_name, self.correlation_cache_correlation_method)
        section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)

        if debug_mode:
            # audio section
            print("audio_section length",len(audio_section))
        # Cross-correlate and normalize correlation
        correlation = correlate(audio_section, clip, mode='full', method='fft')
        if debug_mode:
            print("correlation length", len(correlation))

        # abs
        correlation = np.abs(correlation)
        correlation /= np.max(correlation)

        max_index = np.argmax(correlation)

        # slice
        correlation = slicing_with_zero_padding(correlation, len(correlation_clip), max_index)

        similarity = calculate_similarity(correlation_clip,correlation)

        if debug_mode:
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

            print(f"{section_ts} similarity",similarity)

            debug_dir = f"./tmp/debug/non_repeating_cross_correlation_{clip_name}"
            os.makedirs(debug_dir, exist_ok=True)
            print(json.dumps({"max_index":max_index,
                              "similarity":similarity,
                              }, indent=2, cls=NumpyEncoder),
                  file=open(f'{debug_dir}/{clip_name}_{index}_{section_ts}.txt', 'w'))

        similarity_threshold = 0.002

        if similarity > similarity_threshold:
            if debug_mode:
                print(f"failed verification for {section_ts} due to similarity {similarity} > {similarity_threshold}")
            return []
        else:
            return [max_index / sr]
