import argparse
import datetime
import librosa
import numpy as np
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

import ffmpeg
import librosa
import soundfile as sf

#import pyaudio

'''
use ffmpeg steaming, which supports more format for streaming
'''

def load_audio_file(file_path, sr=None):
    return librosa.load(file_path, sr=sr, mono=True)  # mono=True ensures a single channel audio

def process_chunk(chunk, clip, sr, threshold, previous_chunk):
    # Concatenate previous chunk for continuity in processing
    audio_section = np.concatenate((previous_chunk, chunk))
    
    # Normalize the current chunk
    audio_section = audio_section / np.max(np.abs(audio_section))
    
    # Cross-correlate and normalize correlation
    correlation = correlate(audio_section, clip, mode='valid')
    correlation = np.abs(correlation)
    correlation /= np.max(correlation)

    # Detect if there are peaks exceeding the threshold
    peaks = np.where(correlation > threshold)[0]
    peak_times = (peaks + 1) / sr

    prev_seconds = len(previous_chunk)/sr
    print(f"prev_seconds: {prev_seconds}")
    #print(f"peak_times: {peak_times}")

    # filter out those from previous chunk
    peak_times_final = [peak_time for peak_time in peak_times if peak_time > prev_seconds]

    return peak_times_final, correlation

def find_clip_in_audio_in_chunks(clip_path, full_audio_path, chunk_duration=10):
    target_sample_rate = 16000

    # Load the audio clip
    clip, sr_clip = load_audio_file(clip_path,sr=target_sample_rate) # 16k

    # Normalize the clip
    clip = clip / np.max(np.abs(clip))

    # Initialize parameters
    threshold = 0.8  # Threshold for distinguishing peaks
    previous_chunk = np.zeros(0)  # Buffer to maintain continuity between chunks
    #print("previous_chunk")
    #print(previous_chunk)
    #print(len(previous_chunk))
    all_peak_times = []
    all_correlation = []

    # Create ffmpeg process
    process = (
        ffmpeg
        .input(full_audio_path)
        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sample_rate, loglevel="quiet")
        .run_async(pipe_stdout=True)
    )

    seconds_per_chunk = 60

    # for streaming
    frame_length = (seconds_per_chunk * sr_clip) * 2  # times two because it is 2 bytes per sample
    chunk_size=frame_length
    i = 0
    # Process audio in chunks
    while True:
        in_bytes = process.stdout.read(chunk_size)
        if not in_bytes:
            break
        # Convert bytes to numpy array
        chunk = np.frombuffer(in_bytes, dtype="int16")
        #sf.write(f"./tmp/sound{i}.wav", chunk, target_sample_rate)
        #print("chunk....")
        #print(len(chunk))
        #exit(1)
        # Process audio data with Librosa (e.g., feature extraction)
        # ... your Librosa processing here ...
        peak_times, correlation = process_chunk(chunk, clip, sr_clip, threshold, previous_chunk)
        if len(peak_times):
            peak_times_from_beginning = [time + (i*seconds_per_chunk) for time in peak_times]
            #print(f"Found occurrences at: {peak_times_from_beginning} seconds, chunk {i}")
            all_peak_times.extend(peak_times_from_beginning)
            all_correlation.extend(correlation)
        
        # Update previous_chunk to current chunk
        previous_chunk = chunk
        i = i + 1

    process.wait()

    return all_peak_times, all_correlation


# def find_clip_in_audio_in_chunks2(clip_path, full_audio_path, chunk_duration=10):
#     target_sample_rate = 16000

#     # Load the audio clip
#     clip, sr_clip = load_audio_file(clip_path, sr=target_sample_rate)

#     # Check if sampling rates match, resample if necessary
#     if sr_clip != target_sample_rate:
#         raise "mismatch"

#     # Write the audio data to a new WAV file
#     sf.write("./tmp/test.wav", clip, target_sample_rate)

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
#             #sf.write(f"./tmp/sound{i}.wav", audio_data, target_sample_rate)

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
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()


    # Find clip occurrences in the full audio
    peak_times, correlation = find_clip_in_audio_in_chunks(args.pattern_file, args.audio_file)

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
