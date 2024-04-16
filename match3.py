import argparse
import datetime
import librosa
import numpy as np
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

import ffmpeg
import librosa
#import pyaudio

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

    return peak_times, correlation

def find_clip_in_audio_in_chunks(clip_path, full_audio_path, chunk_duration=10):
    target_sample_rate = 16000

    # Load the audio clip
    clip, sr_clip = load_audio_file(clip_path,sr=target_sample_rate) # 16k

    # Normalize the clip
    clip = clip / np.max(np.abs(clip))

    # Initialize parameters
    threshold = 0.8  # Threshold for distinguishing peaks
    previous_chunk = np.zeros_like(clip)  # Buffer to maintain continuity between chunks
    
    all_peak_times = []
    all_correlation = []

    # Create ffmpeg process
    process = (
        ffmpeg
        .input(full_audio_path)
        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sample_rate)
        .run_async(pipe_stdout=True)
    )

    # for streaming
    frame_length = (2048 * sr_clip)
    chunk_size=frame_length

    # Process audio in chunks
    while True:
        in_bytes = process.stdout.read(chunk_size)
        if not in_bytes:
            break
        # Convert bytes to numpy array
        chunk = np.frombuffer(in_bytes, dtype="int16")
        # Process audio data with Librosa (e.g., feature extraction)
        # ... your Librosa processing here ...
        peak_times, correlation = process_chunk(chunk, clip, sr_clip, threshold, previous_chunk)
        if len(peak_times):
            print(f"Found occurrences at: {peak_times} seconds")
            all_peak_times.extend(peak_times)
            all_correlation.extend(correlation)
        
        # Update previous_chunk to current chunk
        previous_chunk = chunk

    process.wait()

    return all_peak_times, all_correlation




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
    

    # Optional: plot the correlation graph to visualize
    plt.figure(figsize=(10, 4))
    plt.plot(correlation)
    plt.title('Cross-correlation between the audio clip and full track')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coefficient')
    plt.savefig('./tmp/cross_correlation2.png')

if __name__ == '__main__':
    main()
