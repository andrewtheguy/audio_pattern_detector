import argparse
import datetime
import librosa
import numpy as np
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

'''
use librosa with example of chunking
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
    peak_times = (peaks) / sr

    return peak_times, correlation

# only works for wav files
def find_clip_in_audio_in_chunks(clip_path, full_audio_path, chunk_duration=10):
    # Load the audio clip
    clip, sr_clip = load_audio_file(clip_path)

    # Normalize the clip
    clip = clip / np.max(np.abs(clip))

    # Initialize parameters
    threshold = 0.8  # Threshold for distinguishing peaks
    previous_chunk = np.zeros(0)  # Buffer to maintain continuity between chunks
    
    all_peak_times = []
    all_correlation = []

    # Set the frame parameters to be equivalent to the librosa defaults
    # in the file's native sampling rate
    frame_length = (60 * sr_clip)
    hop_length = (30 * sr_clip)

    # Stream over the full audio in chunks
    for i, chunk in enumerate(librosa.stream(full_audio_path, block_length=chunk_duration, frame_length=frame_length, hop_length=hop_length)):
        peak_times, correlation = process_chunk(chunk, clip, sr_clip, threshold, previous_chunk)
        
        if len(peak_times):
            print(f"Found occurrences in chunk {i+1}")
            all_peak_times.extend([peak_time + i * chunk_duration for peak_time in peak_times])
            all_correlation.extend(correlation)
        
        # Update previous_chunk to current chunk
        previous_chunk = chunk

    return all_peak_times, all_correlation


def find_clip_in_audio(clip_path, full_audio_path):
    # Load the audio clip and the full audio
    clip, sr_clip = librosa.load(clip_path, sr=None)
    audio, sr_audio = librosa.load(full_audio_path, sr=None)
    
    # Check if sampling rates match, resample if necessary
    if sr_clip != sr_audio:
        clip = librosa.resample(clip, orig_sr=sr_clip, target_sr=sr_audio)
    
    # Normalizing the audio files may help with cross-correlation
    clip = clip / np.max(np.abs(clip))
    audio = audio / np.max(np.abs(audio))
    
    # Cross-correlate the full audio with the clip
    correlation = correlate(audio, clip, mode='full')
    correlation = np.abs(correlation)
    correlation /= np.max(correlation)  # Normalize correlation
    
    # Find points where the correlation is high
    threshold = 0.8  # Threshold for peak detection, you may need to adjust this
    peaks = np.where(correlation > threshold)[0]
    #peaks = np.where(correlation >= threshold)[0]
    
    # Convert peak indices to times
    peak_times = (peaks - len(clip) + 1) / sr_audio
    
    return peak_times, correlation


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
