import os
import sys

import soundfile as sf

from scipy.signal import find_peaks
from scipy import signal

from audio_pattern_detector.detection_utils import is_pure_tone


def is_news_report_beep(audio_data, sample_rate, graph_file_name="test"):
    """
    Determine if the given audio data represents a pure tone.

    Parameters:
        audio_data (numpy array): The audio data as a floating-point array.
        sample_rate (int): The sample rate of the audio data in Hz.

    Returns:
        bool: True if the audio is a pure tone, False otherwise.
    """

    #audio_data = audio_data[0:3662]

    # Perform FFT
    fft_result = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(audio_data), d=1 / sample_rate)

    # Analyze the magnitude spectrum
    magnitude = np.abs(fft_result)
    positive_freqs = freqs[:len(freqs) // 2]
    positive_magnitude = magnitude[:len(freqs) // 2]

    graph_dir = f"./tmp/graph/pure_tone"
    os.makedirs(graph_dir, exist_ok=True)

    plt.plot(positive_freqs, positive_magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')
    # plt.show()
    plt.savefig(
        f'{graph_dir}/{graph_file_name}.png')
    plt.close()

    # Find the dominant frequency
    dominant_freq_idx = np.argmax(positive_magnitude)
    dominant_magnitude = positive_magnitude[dominant_freq_idx]

    #print(f"positive_magnitude: {positive_magnitude}")

    # # Define a threshold for pure tone
    # noise_threshold = 0.25 * dominant_magnitude  # e.g., 25% of the peak
    # peaks = positive_magnitude > noise_threshold
    #
    # #print(f"dominant_freq: {positive_freqs[dominant_freq_idx]}")
    # #print(f"peaks magnitudes: {positive_magnitude[peaks]}")
    #
    # peak_freqs = positive_freqs[peaks]
    # peak_magnitudes = positive_magnitude[peaks]
    # #dominant_freq = positive_freqs[dominant_freq_idx]
    #
    # news_report_freq_range = np.where((peak_freqs >= 1030) | (peak_freqs <= 1042))
    #
    # neighboring_range_left = np.where((peak_freqs >= 1010) | (peak_freqs < 1030))
    # neighboring_range_right = np.where((peak_freqs > 1042) | (peak_freqs <= 1060))

    normalized_magitudes = positive_magnitude / dominant_magnitude

    # Peak detection (using scipy.signal.find_peaks):
    peaks, peak_props = find_peaks(normalized_magitudes, prominence=0.4,width=[0,4],rel_height=0.7)  # Adjust prominence

    peak_freqs = positive_freqs[peaks]

    print(f"peaks: {peaks}")
    print(f"peak_freqs: {peak_freqs}")
    print(f"peak_props: {peak_props}")
    #exit(1)

    result = {
        "dominant_freq": positive_freqs[dominant_freq_idx],
        "peaks": peaks,
        "peak_freqs": peak_freqs,
        "peak_props": peak_props,
    }

    if np.any((peak_freqs > 1035) & (peak_freqs < 1042)):
        print("News report detected.")
        result["is_news_report_clip"] = True
    else:
        print("Not a news report clip.")
        result["is_news_report_clip"] = False

    return result

    # result = {
    #     "dominant_freq": positive_freqs[dominant_freq_idx],
    #     "peak_freqs": peak_freqs,
    #     "peak_magnitudes": peak_magnitudes,
    #     "neighboring_range_left": neighboring_range_left,
    #     "neighboring_range_right": neighboring_range_right,
    # }
    #
    # if len(neighboring_range_left) > 0 or len(neighboring_range_right) > 0:
    #     print("Not a news report clip.")
    #     result["is_news_report_clip"] = False
    #     result["fail_reason"] = "neighboring frequency range are prominent enough"
    #
    # if len(news_report_freq_range) == 0:
    #     print("Not a news report clip.")
    #     result["is_news_report_clip"] = False
    #     result["fail_reason"] = "dominant frequency not in range"
    # else:
    #     print("News report clip detected.")
    #     result["is_news_report_clip"] = True

    #return result

    # # Check if there's only one significant peak
    # if np.sum(peaks) == 1:
    #     print(f"Pure tone detected at frequency: {positive_freqs[dominant_freq_idx]:.2f} Hz")
    #     return True
    # else:
    #     print("Not a pure tone.")
    #     return False


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt



def detect_sine_tone(audio_signal, sample_rate=8000, tone_frequency=1039, bandwidth=20):
    # Design a bandpass filter centered at tone_frequency with a specified bandwidth
    nyquist_rate = sample_rate / 2.0
    low_cutoff = (tone_frequency - bandwidth) / nyquist_rate
    high_cutoff = (tone_frequency + bandwidth) / nyquist_rate

    # Create a bandpass filter
    b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')

    # Apply the filter to the audio signal
    filtered_signal = signal.filtfilt(b, a, audio_signal)

    # Perform FFT to analyze frequency content
    fft_signal = np.fft.fft(filtered_signal)
    freqs = np.fft.fftfreq(len(fft_signal), 1 / sample_rate)

    # Find the index corresponding to the target frequency (1039 Hz)
    target_index = np.argmin(np.abs(freqs - tone_frequency))

    # Find the magnitude of the tone at the target frequency
    tone_magnitude = np.abs(fft_signal[target_index])

    freqs2 = freqs[:len(freqs) // 2]
    signal2 = np.abs(fft_signal)[:len(freqs) // 2]
    signal2 = signal2 / np.max(signal2)  # Normalize the magnitude


    peaks = find_peaks(signal2,threshold=0,width=[0,4],prominence=0.2,rel_height=0.5)
    # # print("freqs2: ", freqs2)
    # # print("signal2: ", signal2)
    # # print("tone_magnitude: ", tone_magnitude)
    # print("peaks: ", peaks)
    # #print("len(peaks[0]): ", len(peaks[0]))
    #
    # # Visualize the frequency spectrum of the filtered signal
    plt.figure(figsize=(10, 6))
    plt.plot(freqs2,signal2)
    plt.title(f'Frequency Spectrum around {tone_frequency} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 2000)  # Limiting x-axis for better clarity
    plt.show()

    # needs to be high enough
    is_news_report_clip = bool(len(peaks[0]) == 1 and peaks[1]["prominences"][0] > 0.9)

    # should return true only if there is one sharp peak
    return {
        "is_news_report_clip":is_news_report_clip,
        "tone_magnitude":tone_magnitude,
        "peaks":peaks
    }

def plot_spectrogram(audio_data, sample_rate, title='Spectrogram'):
    """
    Generate and plot a spectrogram of the audio data

    Parameters:
    -----------
    audio_data : numpy.ndarray
        Raw audio data as a 1D float array
    sample_rate : int
        Sampling rate of the audio
    title : str, optional
        Title for the spectrogram plot
    """
    plt.figure(figsize=(12, 8))

    # Create subplots
    plt.subplot(3, 1, 1)
    plt.title('Original Signal')
    plt.plot(np.linspace(0, len(audio_data) / sample_rate, len(audio_data)), audio_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(audio_data, fs=sample_rate, nperseg=256)

    plt.subplot(3, 1, 2)
    plt.title(title)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')

    # Compute and plot power spectrum
    frequencies, power_spectrum = signal.periodogram(audio_data, fs=sample_rate)

    plt.subplot(3, 1, 3)
    plt.title('Power Spectrum')
    plt.semilogy(frequencies, power_spectrum)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density')
    plt.xlim(0, sample_rate / 2)  # Nyquist frequency

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # # Example usage
    # # Simulate raw audio data of a 400 Hz sine wave
    # sample_rate = 8000
    # duration = 1.0  # seconds
    # frequency = 400  # Hz
    # time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # audio_data = 0.5 * np.sin(2 * np.pi * frequency * time)
    #
    # # Check if it's a pure tone
    # is_pure_tone(audio_data, sample_rate)

    filename = sys.argv[1]
    # Read the WAV file
    data, samplerate = sf.read(filename)

    # Check if the audio is mono
    if len(data.shape) > 1 and data.shape[1] != 1:
        raise ValueError("The file is not mono.")

    # The audio data is already a float array if `soundfile` loads it directly.
    print("Sample rate:", samplerate)


    is_pure_tone(data, samplerate)

    #result = is_news_report_beep(data, samplerate)
    #print(result)

    #print(detect_sine_tone(data, samplerate))
    plot_spectrogram(data, samplerate)



