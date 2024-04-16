import argparse
import datetime
import librosa
import numpy as np
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

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
    peak_times, correlation = find_clip_in_audio(args.pattern_file, args.audio_file)

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
