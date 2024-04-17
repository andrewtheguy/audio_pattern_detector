import librosa
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
import dtw

target_sr=16000

def match_audio_mal(clip_path, audio_path):
    hop_length = 512  # Ensure this matches the hop_length used for Mel Spectrogram
    method = "euclidean"

    # Load audio files and extract MFCCs (similar to previous example)
    # ...
    clip, sr1 = librosa.load(clip_path,sr=target_sr,mono=True)
    audio, sr2 = librosa.load(audio_path,sr=target_sr,mono=True)

    # Extract Mel Spectrograms
    clip_melspec = librosa.feature.melspectrogram(y=clip, sr=sr1, hop_length=hop_length)
    audio_melspec = librosa.feature.melspectrogram(y=audio, sr=sr1, hop_length=hop_length)

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
        plt.savefig('./tmp/distances.png')

        # Find minimum distance and its index
        min_distance = min(distances)
        match_index = np.argmin(distances)

    elif method == "dtw":
        # Calculate DTW distance
        d, wp = dtw(clip_melspec.T, audio_melspec.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

        # Get the ending frame of the match in the larger audio
        match_index = wp[-1, 1] - clip_melspec.shape[1] + 1

    else:
        raise ValueError("Invalid method. Choose 'euclidean' or 'dtw'")

    # Convert match index to timestamp
    
    match_time = (match_index * hop_length) / sr2  

    return match_time, min_distance if method == "euclidean" else d.distance


def match_audio_euclidean(clip_path, audio_path):
    hop_length = 512  # Adjust hop_length as used in MFCC extraction
    # Load audio files
    clip, sr1 = librosa.load(clip_path,sr=target_sr,mono=True,hop_length=hop_length)
    audio, sr2 = librosa.load(audio_path,sr=target_sr,mono=True,hop_length=hop_length)

    # Ensure same sample rate
    if sr1 != sr2:
        audio = librosa.resample(audio, sr2, sr1)

    # Extract MFCC features
    clip_mfcc = librosa.feature.mfcc(y=clip, sr=sr1)
    audio_mfcc = librosa.feature.mfcc(y=audio, sr=sr1)

    
    distances = []
    for i in range(audio_mfcc.shape[1] - clip_mfcc.shape[1] + 1):
        dist = np.linalg.norm(clip_mfcc - audio_mfcc[:, i:i+clip_mfcc.shape[1]])
        distances.append(dist)

    # Find minimum distance and its index
    match_index = np.argmin(distances)
    min_distance = distances[match_index]

    # Optional: plot the two MFCC sequences
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Main Audio MFCC')
    plt.imshow(audio_mfcc.T, aspect='auto', origin='lower')
    plt.subplot(1, 2, 2)
    plt.title('Pattern Audio MFCC')
    plt.imshow(clip_mfcc.T, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.savefig(f'./tmp/MFCC.png')
    plt.close()

    distances_ratio = [dist / min_distance for dist in distances if dist / min_distance <= 2]

    # Optional: plot the correlation graph to visualize
    plt.figure(figsize=(80,40))
    plt.plot(distances_ratio)
    plt.title('distances')
    plt.xlabel('index')
    plt.ylabel('distance')
    plt.savefig('./tmp/distances.png')

    distances_selected = np.where(distances / min_distance <= 2)[0]


    # Convert match index to timestamp
    match_times = (distances_selected * hop_length) / sr2  # sr2 is the sampling rate of audio


    return match_times, min_distance


def main():
    # Load and process both audio files
    main_audio_path = './tmp/audio_section2.wav'
    pattern_audio_path = './tmp/inputs/rthk_beep.wav'
    #match_times, min_distance = match_audio_euclidean(pattern_audio_path, main_audio_path)
    #print(match_times,min_distance)
    match_times, min_distance = match_audio_mal(pattern_audio_path, main_audio_path)
    print(match_times,min_distance)

if __name__ == '__main__':
    main()
