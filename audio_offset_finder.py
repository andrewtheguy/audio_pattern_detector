import argparse
import datetime

import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def find_offset(audio_file, pattern_file, window):
    y_audio, sr_audio = librosa.load(audio_file, sr=None)
    y_pattern, _ = librosa.load(pattern_file, sr=sr_audio)

    c = signal.correlate(y_audio, y_pattern[:sr_audio*window], mode='valid', method='fft')
    peak = np.argmax(c)
    offset = round(peak / sr_audio, 2)

    fig, ax = plt.subplots()
    ax.plot(c)
    fig.savefig("./tmp/cross-correlation.png")

    return offset



def recognize_sound_pattern(audio_file, pattern_file):
    # Load the audio clip and the pattern audio
    audio, sr = librosa.load(audio_file)
    pattern, _ = librosa.load(pattern_file)

    # Extract features from the audio clip and the pattern
    audio_features = librosa.feature.melspectrogram(y=audio, sr=sr)
    pattern_features = librosa.feature.melspectrogram(y=pattern, sr=sr)

    # Compute the similarity matrix between the audio features and the pattern features
    similarity_matrix = librosa.segment.cross_similarity(audio_features, pattern_features, mode='distance')

    # Find the indices of the maximum similarity values
    indices = np.argmax(similarity_matrix, axis=1)

    # Get the corresponding time stamps of the matched patterns
    time_stamps = librosa.frames_to_time(indices, sr=sr)

    return time_stamps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-file', metavar='pattern file', type=str, help='pattern file')
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')
    parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of a target audio')
    args = parser.parse_args()

    #matched_time_stamps = recognize_sound_pattern(args.audio_file, args.pattern_file)

    #print("Sound pattern found at the following time stamps:")
    #for time_stamp in matched_time_stamps:
    #    print(f"{str(datetime.timedelta(seconds=time_stamp))} seconds")
    offset = find_offset(args.audio_file, args.pattern_file, args.window)
    print(f"Offset: {offset}s" )
    print(f"ts: {str(datetime.timedelta(seconds=offset))}" )

if __name__ == '__main__':
    main()