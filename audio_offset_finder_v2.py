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

logger = logging.getLogger(__name__)

#ignore possible clipping
warnings.filterwarnings('ignore', module='pyloudnorm')

'''
use ffmpeg steaming, which supports more format for streaming
'''

target_sample_rate = 8000


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


# sample rate needs to be the same for both or bugs will happen
def chroma_method(clip, audio, sr):
    #global method_count
    # Extract features from the audio clip and the pattern
    audio_features = librosa.feature.chroma_cqt(y=audio, sr=sr)
    pattern_features = librosa.feature.chroma_cqt(y=clip, sr=sr)

    # Compute the similarity matrix between the audio features and the pattern features
    similarity_matrix = librosa.segment.cross_similarity(audio_features, pattern_features, mode='distance')

    # Find the indices of the maximum similarity values
    indices = np.argmax(similarity_matrix, axis=1)

    # Get the corresponding time stamps of the matched patterns
    time_stamps = librosa.frames_to_time(indices, sr=sr)
    #method_count = method_count + 1
    return time_stamps


# sample rate needs to be the same for both or bugs will happen
def mfcc_method(clip, audio, sr, index, seconds_per_chunk, clip_name):
    # Extract features from the audio clip and the pattern
    audio_features = librosa.feature.mfcc(y=audio, sr=sr)
    pattern_features = librosa.feature.mfcc(y=clip, sr=sr)

    # Compute the similarity matrix between the audio features and the pattern features
    similarity_matrix = librosa.segment.cross_similarity(audio_features, pattern_features, mode='distance')

    # Find the indices of the maximum similarity values
    indices = np.argmax(similarity_matrix, axis=1)

    # Get the corresponding time stamps of the matched patterns
    time_stamps = librosa.frames_to_time(indices, sr=sr)

    return time_stamps


# sample rate needs to be the same for both or bugs will happen
def mfcc_method2(clip, audio, sr, index, seconds_per_chunk, clip_name):
    frame = len(clip)
    hop_length = 512  # Ensure this matches the hop_length used for Mel Spectrogram

    # Extract MFCC features
    clip_mfcc = librosa.feature.mfcc(y=clip, sr=sr, hop_length=hop_length)
    audio_mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=hop_length)

    distances = []
    for i in range(audio_mfcc.shape[1] - clip_mfcc.shape[1] + 1):
        dist = np.linalg.norm(clip_mfcc - audio_mfcc[:, i:i + clip_mfcc.shape[1]])
        distances.append(dist)

    # Find minimum distance and its index
    match_index = np.argmin(distances)
    min_distance = distances[match_index]

    # # Optional: plot the two MFCC sequences
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.title('Main Audio MFCC')
    # plt.imshow(audio_mfcc.T, aspect='auto', origin='lower')
    # plt.subplot(1, 2, 2)
    # plt.title('Pattern Audio MFCC')
    # plt.imshow(clip_mfcc.T, aspect='auto', origin='lower')
    # plt.tight_layout()
    # plt.savefig(f'./tmp/MFCC.png')
    # plt.close()

    #distances_ratio = [dist / min_distance for dist in distances]

    os.makedirs("./tmp/graph/mfcc", exist_ok=True)
    #Optional: plot the correlation graph to visualize
    plt.figure(figsize=(20, 8))
    plt.plot(distances)
    plt.title('Cross-correlation between the audio clip and full track')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coefficient')
    plt.savefig(
        f'./tmp/graph/mfcc/distance_{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.png')
    plt.close()

    distances_selected = np.where(distances / min_distance <= 1.05)[0]

    # Convert match index to timestamp
    match_times = (distances_selected * hop_length) / sr  # sr is the sampling rate of audio

    return match_times


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


def find_outliers(data):
    mean = np.mean(data)
    std = np.std(data)

    threshold = 3
    outliers = []
    for x in data:
        z_score = (x - mean) / std
        if abs(z_score) > threshold:
            outliers.append(x)
    return outliers


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


max_test = []


def advanced_correlation_method(clip, audio, sr, index, seconds_per_chunk, clip_name):
    global max_test
    #threshold = 0.7  # Threshold for distinguishing peaks, need to be smaller for larger clips
    # Cross-correlate and normalize correlation
    correlation = correlate(audio, clip, mode='full', method='fft')
    correlation = np.abs(correlation)
    # -1 to bump the max to be above 1
    correlation /= np.max(correlation) - 1

    #correlation = downsample(int(sr/10),correlation)

    section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)
    graph_dir = f"./tmp/graph/cross_correlation_{clip_name}"
    os.makedirs(graph_dir, exist_ok=True)
    # Optional: plot the correlation graph to visualize
    plt.figure(figsize=(10, 4))
    plt.plot(correlation)
    plt.title('Cross-correlation between the audio clip and full track')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coefficient')
    plt.savefig(f'{graph_dir}/{index}_{section_ts}.png')
    plt.close()

    peak_dir = f"./tmp/peaks/cross_correlation_{clip_name}"
    os.makedirs(peak_dir, exist_ok=True)
    percentile=np.percentile(correlation, 95)
    print(f"np.percentile(correlation, 95) for {section_ts}",percentile)

    height = 0.7
    # find the peaks in the spectrogram
    peaks, properties = find_peaks(correlation, threshold=percentile, height=height)

    print(json.dumps(peaks.tolist(), indent=2), file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))

    # outliers = compute_mod_z_score(correlation)
    #
    # graph_dir = f"./tmp/graph/outliers_{clip_name}"
    # os.makedirs(graph_dir, exist_ok=True)
    # # Optional: plot the correlation graph to visualize
    # plt.figure(figsize=(10, 4))
    # plt.plot(outliers)
    # plt.title('Cross-correlation outliers between the audio clip and full track')
    # plt.xlabel('outliers')
    # plt.ylabel('Correlation coefficient')
    # plt.savefig(f'{graph_dir}/{index}_{section_ts}.png')
    # plt.close()
    #
    # # peak_max = np.max(correlation)
    # # index_max = np.argmax(correlation)
    #
    # max_score = np.max(outliers)
    # max_test.append(max_score)
    # print(f"max_score for {clip_name} {section_ts}: {max_score}")

    peaks_final=[]
    # not trying to match many occurences
    if percentile < 0.05:
        peaks_final.extend(peaks)
    peak_times = np.array(np.asarray(peaks_final)) / sr

    return peak_times


def correlation_method(clip, audio, sr, index, seconds_per_chunk, clip_name):
    threshold = 0.7  # Threshold for distinguishing peaks, need to be smaller for larger clips
    # Cross-correlate and normalize correlation
    correlation = correlate(audio, clip, mode='full', method='fft')
    correlation = np.abs(correlation)
    # -1 to bump the max to be above 1
    correlation /= np.max(correlation) - 1

    os.makedirs("./tmp/graph", exist_ok=True)
    #Optional: plot the correlation graph to visualize
    plt.figure(figsize=(10, 4))
    plt.plot(correlation)
    plt.title('Cross-correlation between the audio clip and full track')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coefficient')
    plt.savefig(
        f'./tmp/graph/cross_correlation_{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.png')
    plt.close()

    peaks = np.where(correlation > threshold)[0]

    peak_times = np.array(peaks) / sr

    return peak_times


# sliding_window: for previous_chunk in seconds from end
# index: for debugging by saving a file for audio_section
# seconds_per_chunk: default seconds_per_chunk
def process_chunk(chunk, clip, sr, previous_chunk, sliding_window, index, seconds_per_chunk, clip_name,
                  method="correlation"):
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
            audio_section = np.concatenate((previous_chunk[(-sliding_window * sr):], chunk, np.array([])))
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

    samples_skip_end = 0
    # needed for correlation method
    if method == "correlation":
        audio_section = np.concatenate((audio_section, clip))
        samples_skip_end = clip_length

    os.makedirs("./tmp/audio", exist_ok=True)
    sf.write(
        f"./tmp/audio/section_{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.wav",
        audio_section, sr)

    if method == "correlation":
        peak_times = correlation_method(clip, audio=audio_section, sr=sr, index=index,
                                        seconds_per_chunk=seconds_per_chunk, clip_name=clip_name)
    elif method == "advanced_correlation":
        peak_times = advanced_correlation_method(clip, audio=audio_section, sr=sr, index=index,
                                        seconds_per_chunk=seconds_per_chunk, clip_name=clip_name)
    elif method == "mfcc":
        peak_times = mfcc_method(clip, audio=audio_section, sr=sr, index=index, seconds_per_chunk=seconds_per_chunk,
                                 clip_name=clip_name)
    elif method == "chroma_method":
        peak_times = chroma_method(clip, audio=audio_section, sr=sr)
    else:
        raise ValueError("unknown method")

    peak_times2 = []
    for t in peak_times:
        if t >= 0 and t >= (len(audio_section) - samples_skip_end - 1) / sr:
            #skip the placeholder clip at the end
            continue
        peak_times2.append(t)

    peak_times = peak_times2
    peak_times_final = [peak_time - subtract_seconds for peak_time in peak_times]
    #peak_times_final = [peak_time for peak_time in peak_times_final if peak_time >= 0]

    #for item in correlation:
    #    if item > threshold:
    #        print(item)
    return peak_times_final


def cleanup_peak_times(peak_times):
    # freq = {}

    # for peak in peak_times:
    #     i = math.floor(peak)
    #     cur = freq.get(i, 0)
    #     freq[i] = cur + 1

    #print(freq)

    #print({k: v for k, v in sorted(freq.items(), key=lambda item: item[1])})

    #print('before consolidate',peak_times)

    # deduplicate
    peak_times_clean = list(dict.fromkeys([math.floor(peak) for peak in peak_times]))

    peak_times_clean2 = deque(sorted(peak_times_clean))
    #print('before remove close',peak_times_clean2)

    peak_times_final = []

    # skip those less than 10 seconds in between like beep, beep, beep
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
    # Convert buffer to float32 using NumPy                                                                                 
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0                                                      
    max_int16 = 2 ** 15
    audio_normalised = audio_as_np_float32 / max_int16
    return audio_normalised


def convert_audio_to_clip_format(audio_path, output_path):
    # Load the audio clip
    clip = load_audio_file(audio_path, sr=target_sample_rate)

    # convert to float
    clip = convert_audio_arr_to_float(clip)

    sf.write(output_path, clip, target_sample_rate)


# unwind_clip_ts for starting timestamps like intro
# could cause issues with small overlap when intro is followed right by news report
def find_clip_in_audio_in_chunks(clip_path, full_audio_path, method="correlation"):
    global max_test
    max_test = []
    unwind_clip_ts = True

    # Load the audio clip
    clip = load_audio_file(clip_path, sr=target_sample_rate)

    # convert to float
    clip = convert_audio_arr_to_float(clip)

    #print(clip)

    #zeroes = np.zeros(int(1 * target_sample_rate))

    #clip = np.concatenate((zeroes,clip,zeroes))

    #clip = clip / 3

    #sf.write(f"./tmp/clip.wav", clip, target_sample_rate)

    # Initialize parameters

    previous_chunk = None  # Buffer to maintain continuity between chunks
    #print("previous_chunk")
    #print(previous_chunk)
    #print(len(previous_chunk))
    all_peak_times = []

    # Create ffmpeg process
    process = (
        ffmpeg
        .input(full_audio_path)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=target_sample_rate, loglevel="error")
        .run_async(pipe_stdout=True)
    )

    seconds_per_chunk = 60
    sliding_window = 5

    clips_length_milis = len(clip) / target_sample_rate

    clip_seconds = int(clips_length_milis)

    if (sliding_window < clip_seconds + 5):
        # need to extend the sliding window to overlap the clip
        sliding_window = clip_seconds + 5
        print(f"adjusted sliding_window to {sliding_window}")
    #print(sliding_window)

    if (seconds_per_chunk < sliding_window * 2):
        seconds_per_chunk = sliding_window * 10
        print(f"adjusted seconds_per_chunk to {seconds_per_chunk}")
    #print(seconds_per_chunk)

    #exit(1)

    audio_size = 0

    # for streaming
    frame_length = (seconds_per_chunk * target_sample_rate)
    chunk_size = frame_length * 2  # times two because it is 2 bytes per sample (int16)
    i = 0
    # Process audio in chunks
    while True:
        in_bytes = process.stdout.read(chunk_size)
        if not in_bytes:
            break
        # Convert bytes to numpy array
        chunk = np.frombuffer(in_bytes, dtype="int16")
        # convert to float 
        chunk = convert_audio_arr_to_float(chunk)

        audio_size += len(chunk)

        #sf.write(f"./tmp/sound{i}.wav", chunk, target_sample_rate)
        #print("chunk....")
        #print(len(chunk))
        #exit(1)
        clip_name, _ = os.path.splitext(os.path.basename(clip_path))
        peak_times = process_chunk(chunk=chunk, clip=clip, sr=target_sample_rate,
                                   previous_chunk=previous_chunk,
                                   sliding_window=sliding_window,
                                   index=i,
                                   clip_name=clip_name,
                                   seconds_per_chunk=seconds_per_chunk, method=method)
        if len(peak_times):
            peak_times_from_beginning = [time + (i * seconds_per_chunk) for time in peak_times]
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

        # Update previous_chunk to current chunk
        previous_chunk = chunk
        i = i + 1

    process.wait()

    if(method == "advanced_correlation"):
        #Optional: plot the correlation graph to visualize
        graph_dir = f"./tmp/graph/max_score_{clip_name}"
        os.makedirs(graph_dir, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(max_test)
        plt.title('max outliers between the audio clip and full track')
        plt.xlabel('outliers')
        plt.ylabel('Correlation coefficient')
        plt.savefig(f'{graph_dir}/{clip_name}.png')
        plt.close()

    peak_times_clean = cleanup_peak_times(all_peak_times)
    return peak_times_clean
