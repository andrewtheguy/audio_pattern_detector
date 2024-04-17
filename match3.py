import argparse
import copy
import datetime
import time
import librosa
import numpy as np
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

import ffmpeg
import librosa
import soundfile as sf

#import pyaudio

method_count = 0

'''
use ffmpeg steaming, which supports more format for streaming
'''

def load_audio_file(file_path, sr=None):
    return librosa.load(file_path, sr=sr, mono=True)  # mono=True ensures a single channel audio


def melspectrogram_method(clip,audio,sr):
    global method_count
    frame = len(clip)
    hop_length = 512  # Ensure this matches the hop_length used for Mel Spectrogram
    method = "euclidean"

    # Extract Mel Spectrograms
    clip_melspec = librosa.feature.melspectrogram(y=clip, sr=sr, hop_length=hop_length)
    audio_melspec = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length)

    # Amplitude to dB conversion
    clip_melspec = librosa.power_to_db(clip_melspec)
    audio_melspec = librosa.power_to_db(audio_melspec)

    if method == "euclidean":
        # Calculate Euclidean distance for each frame
        distances = []
        for i in range(audio_melspec.shape[1] - clip_melspec.shape[1] + 1):
            dist = np.linalg.norm(clip_melspec - audio_melspec[:, i:i+clip_melspec.shape[1]])
            distances.append(dist)
        
        # Optional: plot the correlation graph to visualize
        plt.figure(figsize=(20,10))
        plt.plot(distances)
        plt.title('distances')
        plt.xlabel('index')
        plt.ylabel('distance')
        plt.savefig(f'./tmp/melspectrogram_method{method_count}.png')

        # Find minimum distance and its index
        min_distance = min(distances)
        max_distance = max(distances)
        print("min_distance",min_distance)
        print("max_distance",max_distance)
        match_index = np.argmin(distances)
        if match_index:
            print("previous distance") 
            print(distances[match_index-1]) 
            print("end previous distance") 
            print("next distance") 
            print(distances[match_index+1]) 
            print("endnext distance") 
        
        distances_selected = np.where(distances / min_distance <= 1.0)[0]


        # Convert match index to timestamp
        match_times = (distances_selected * hop_length) / sr  # sr is the sampling rate of audio

        method_count = method_count + 1
        return match_times
    # elif method == "dtw":
    #     # Calculate DTW distance
    #     d, wp = dtw(clip_melspec.T, audio_melspec.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

    #     # Get the ending frame of the match in the larger audio
    #     match_index = wp[-1, 1] - clip_melspec.shape[1] + 1

    else:
        raise ValueError("Invalid method. Choose 'euclidean' or 'dtw'")

    # Convert match index to timestamp
    
    #match_time = (match_index * hop_length) / sr2  

    #return match_time, min_distance if method == "euclidean" else d.distance


# sample rate needs to be the same for both or bugs will happen
def mfcc_method(clip,audio,sr):
    global method_count

    frame = len(clip)
    hop_length = 512  # Ensure this matches the hop_length used for Mel Spectrogram

    # Extract MFCC features
    clip_mfcc = librosa.feature.mfcc(y=clip, sr=sr, hop_length=hop_length)
    audio_mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=hop_length)


    distances = []
    for i in range(audio_mfcc.shape[1] - clip_mfcc.shape[1] + 1):
        dist = np.linalg.norm(clip_mfcc - audio_mfcc[:, i:i+clip_mfcc.shape[1]])
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

    # Optional: plot the correlation graph to visualize
    plt.figure(figsize=(20,8))
    plt.plot(distances)
    plt.title('distances')
    plt.xlabel('index')
    plt.ylabel('distance')
    plt.savefig(f'./tmp/mfcc{method_count}.png')

    distances_selected = np.where(distances / min_distance <= 1.0)[0]


    # Convert match index to timestamp
    match_times = (distances_selected * hop_length) / sr  # sr is the sampling rate of audio

    method_count = method_count + 1
    return match_times

def correlation_method(clip,audio,sr):
    threshold = 0.9  # Threshold for distinguishing peaks
    # Cross-correlate and normalize correlation
    correlation = correlate(audio, clip, mode='full',method='direct')
    correlation = np.abs(correlation)
    correlation /= np.max(correlation)

    # # Optional: plot the correlation graph to visualize
    # plt.figure(figsize=(10, 4))
    # plt.plot(correlation)
    # plt.title('Cross-correlation between the audio clip and full track')
    # plt.xlabel('Lag')
    # plt.ylabel('Correlation coefficient')
    # plt.savefig(f'./tmp/cross_correlation{index}.png')
    # plt.close()

    # Detect if there are peaks exceeding the threshold
    peaks = np.where(correlation >= threshold)[0]
    peak_times = peaks / sr
    return peak_times

# sliding_window: for previous_chunk in seconds from end
# index: for debugging by saving a file for audio_section
# seconds_per_chunk: default seconds_per_chunk
def process_chunk(chunk, clip, sr, previous_chunk,sliding_window,index,seconds_per_chunk,method="correlation"):
    new_seconds = len(chunk)/sr
    # Concatenate previous chunk for continuity in processing
    if(previous_chunk is not None):
        if(new_seconds < seconds_per_chunk): # too small
            prev_seconds = len(previous_chunk)/sr
            audio_section = np.concatenate((previous_chunk, chunk))[(-(sliding_window+seconds_per_chunk)*sr):]    
        else:
            prev_seconds = sliding_window
            audio_section = np.concatenate((previous_chunk[(-sliding_window*sr):], chunk))
    else:
        prev_seconds = 0
        audio_section = chunk

    print(f"prev_seconds: {prev_seconds}")
    print(f"new_seconds: {new_seconds}")    

    #sf.write(f"./tmp/audio_section{index}.wav", copy.deepcopy(audio_section), sr)
    # Normalize the current chunk
    audio_section = audio_section / np.max(np.abs(audio_section))
    

    if method == "correlation":
        peak_times = correlation_method(clip, audio=audio_section, sr=sr)
    elif method == "mfcc":
        peak_times = mfcc_method(clip, audio=audio_section, sr=sr)
    elif method == "melspectrogram":
        peak_times = melspectrogram_method(clip, audio=audio_section, sr=sr)
    else:
        raise "unknown method"

    # look back just in case missed something
    peak_times_final = [peak_time - prev_seconds for peak_time in peak_times]
    #for item in correlation:
    #    if item > threshold:
    #        print(item)
    return peak_times_final

def find_clip_in_audio_in_chunks(clip_path, full_audio_path, method="correlation"):
    target_sample_rate = 16000

    # Load the audio clip
    clip, sr_clip = load_audio_file(clip_path,sr=target_sample_rate) # 16k

    # Normalize the clip
    clip = clip / np.max(np.abs(clip))

    #sf.write(f"./tmp/clip.wav", copy.deepcopy(clip), target_sample_rate)

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
        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sample_rate, loglevel="quiet")
        .run_async(pipe_stdout=True)
    )

    seconds_per_chunk = 60
    sliding_window = 60

    # for streaming
    frame_length = (seconds_per_chunk * sr_clip)
    chunk_size=frame_length * 2   # times two because it is 2 bytes per sample
    i = 0
    # Process audio in chunks
    while True:
        in_bytes = process.stdout.read(chunk_size)
        if not in_bytes:
            break
        # Convert bytes to numpy array
        chunk = np.frombuffer(in_bytes, dtype="int16")
        #sf.write(f"./tmp/sound{i}.wav", copy.deepcopy(chunk), target_sample_rate)
        #print("chunk....")
        #print(len(chunk))
        #exit(1)
        # Process audio data with Librosa (e.g., feature extraction)
        # ... your Librosa processing here ...
        peak_times = process_chunk(chunk=chunk, clip=clip, sr=sr_clip, 
                                                previous_chunk=previous_chunk,
                                                sliding_window=sliding_window,index=i,
                                                seconds_per_chunk=seconds_per_chunk, method=method)
        if len(peak_times):
            peak_times_from_beginning = [time + (i*seconds_per_chunk) for time in peak_times]
            #print(f"Found occurrences at: {peak_times} seconds, chunk {i}")
            all_peak_times.extend(peak_times_from_beginning)
            #all_correlation.extend(correlation)
        
        # Update previous_chunk to current chunk
        previous_chunk = chunk
        i = i + 1

    process.wait()

    return all_peak_times


# def find_clip_in_audio_in_chunks2(clip_path, full_audio_path, chunk_duration=10):
#     target_sample_rate = 16000

#     # Load the audio clip
#     clip, sr_clip = load_audio_file(clip_path, sr=target_sample_rate)

#     # Check if sampling rates match, resample if necessary
#     if sr_clip != target_sample_rate:
#         raise "mismatch"

#     # Write the audio data to a new WAV file
#     sf.write("./tmp/test.wav", copy.deepcopy(clip), target_sample_rate)

#     # Normalize the clip
#     clip = clip / np.max(np.abs(clip))

#     # Initialize parameters
#     threshold = 0.8  # Threshold for distinguishing peaks
#     previous_chunk = np.zeros_like(clip)  # Buffer to maintain continuity between chunks
    
#     all_peak_times = []
#     all_correlation = []

#     # Create ffmpeg process
#     process = (
#         ffmpeg
#         .input(full_audio_path)
#         .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sample_rate)
#         .run_async(pipe_stdout=True)
#     )

#     # for streaming
#     seconds_per_chunk = 10
#     #frame_length = (seconds_per_chunk * target_sample_rate)
#     #chunk_size=frame_length

#     # Calculate samples per interval
#     samples_per_interval = int(target_sample_rate * seconds_per_chunk * 2) # times two because it is 2 bytes per sample

#     chunk_size=4096
#     # Process audio in intervals
#     buffer = []

#     i = 0

#     frame_length = (2048 * sr_clip)

#     while True:
#         in_bytes = process.stdout.read(frame_length)
#         if not in_bytes:
#             break
#         buffer.append(np.frombuffer(in_bytes, dtype="int16"))
        
#         if len(buffer) * chunk_size >= samples_per_interval:
#             audio_data = np.concatenate(buffer)
#             # Process 10-second interval audio data with Librosa
#             # Write the audio data to a new WAV file
#             #sf.write(f"./tmp/sound{i}.wav", copy.deepcopy(audio_data), target_sample_rate)

#             #exit(0)    
#             peak_times, correlation = process_chunk(audio_data, clip, target_sample_rate, threshold, previous_chunk)
#             if len(peak_times):
#                 peak_times_from_beginning = [time + (i*seconds_per_chunk) for time in peak_times]
#                 print(f"Found occurrences at: {peak_times_from_beginning} seconds, chunk {i}")
#                 all_peak_times.extend(peak_times_from_beginning)
#                 all_correlation.extend(correlation)

#             # Update previous_chunk to current chunk
#             previous_chunk = audio_data
#             i = i + 1
#             # Clear buffer for next interval
#             buffer = []

#     # Process remaining audio (if any)
#     if buffer:
#         audio_data = np.concatenate(buffer)
#         # ... your Librosa processing here ...
#         peak_times, correlation = process_chunk(audio_data, clip, target_sample_rate, threshold, previous_chunk)
#         if len(peak_times):
#             peak_times_from_beginning = [time + (i*seconds_per_chunk) for time in peak_times]
#             print(f"Found occurrences at: {peak_times_from_beginning} seconds, chunk {i}")
#             all_peak_times.extend(peak_times_from_beginning)
#             all_correlation.extend(correlation)
        
#         # Update previous_chunk to current chunk
#         previous_chunk = audio_data
#         i = i + 1

#     process.wait()

#     return all_peak_times, all_correlation




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-file', metavar='pattern file', type=str, help='pattern file')
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')
    parser.add_argument('--method', metavar='method', type=str, help='correlation,mfcc,melspectrogram',default="correlation")
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()
    #print(args.method)

    # Find clip occurrences in the full audio
    peak_times = find_clip_in_audio_in_chunks(args.pattern_file, args.audio_file, method=args.method)

    peak_times_clean = list(dict.fromkeys([math.floor(peak) for peak in peak_times]))

    #print("Clip occurs at the following times (in seconds):", peak_times_clean)

    for offset in peak_times_clean:
        print(f"Clip occurs at the following times (in seconds): {str(datetime.timedelta(seconds=offset))}" )
    #    #print(f"Offset: {offset}s" )
    

    # # Optional: plot the correlation graph to visualize
    # plt.figure(figsize=(10, 4))
    # plt.plot(correlation)
    # plt.title('Cross-correlation between the audio clip and full track')
    # plt.xlabel('Lag')
    # plt.ylabel('Correlation coefficient')
    # plt.savefig('./tmp/cross_correlation2.png')

if __name__ == '__main__':
    main()
