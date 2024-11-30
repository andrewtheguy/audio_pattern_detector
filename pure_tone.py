import numpy as np
import matplotlib.pyplot as plt


def is_pure_tone(audio_data, sample_rate):
    """
    Determine if the given audio data represents a pure tone.

    Parameters:
        audio_data (numpy array): The audio data as a floating-point array.
        sample_rate (int): The sample rate of the audio data in Hz.

    Returns:
        bool: True if the audio is a pure tone, False otherwise.
    """
    # Perform FFT
    fft_result = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(audio_data), d=1 / sample_rate)

    # Analyze the magnitude spectrum
    magnitude = np.abs(fft_result)
    positive_freqs = freqs[:len(freqs) // 2]
    positive_magnitude = magnitude[:len(freqs) // 2]

    # plt.plot(positive_freqs, positive_magnitude)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('Frequency Spectrum')
    # plt.show()

    # Find the dominant frequency
    dominant_freq_idx = np.argmax(positive_magnitude)
    dominant_magnitude = positive_magnitude[dominant_freq_idx]

    #print(f"dominant_freq_idx: {dominant_freq_idx}")
    #print(f"dominant_magnitude: {dominant_magnitude}")
    print(f"positive_magnitude: {positive_magnitude}")
    #print(f"freqs: {positive_freqs[positive_magnitude]}")

    # Define a threshold for pure tone
    noise_threshold = 0.05 * dominant_magnitude  # e.g., 5% of the peak
    peaks = positive_magnitude > noise_threshold

    print(f"dominant_freq: {positive_freqs[dominant_freq_idx]}")
    print(f"peaks: {positive_magnitude[peaks]}")
    print(f"freqs2: {positive_freqs[peaks]}")

    #indices = np.where(peaks)
    #print(f"freq indices peak: {positive_freqs[indices]}")


    # Check if there's only one significant peak
    if np.sum(peaks) == 1:
        print(f"Pure tone detected at frequency: {positive_freqs[dominant_freq_idx]:.2f} Hz")
        return True
    else:
        print("Not a pure tone.")
        return False


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
