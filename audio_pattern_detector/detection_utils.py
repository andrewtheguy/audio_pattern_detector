import math

import numpy as np
from scipy.integrate import simpson
from scipy.signal import find_peaks


def area_of_overlap_ratio(control, variable):

    if len(control) != len(variable):
        raise ValueError("Both arrays must have the same length")

    total_rect_control = len(control) * max(control)

    y2 = variable
    # Define the x-axis range based on the indices of the input arrays
    x = np.arange(len(control))

    dx=1

    #area_y1 = np.trapz(y1, dx=dx)  # dx=1 since the difference between consecutive x-values is 1
    #area_y2 = np.trapz(y2, dx=dx)  # dx=1 since the difference between consecutive x-values is 1

    area_control = simpson(control, x=x)
    area_y2 = simpson(y2, x=x)

    # To find the overlapping area, take the minimum at each point
    min_curve = np.minimum(control, y2)
    #overlapping_area = np.trapz(min_curve, dx=dx)
    overlapping_area = simpson(min_curve, x=x)

    # Calculate the sum of area of both curves where the two curves don't overlap
    diff_area = area_control+area_y2-2*overlapping_area


    # Calculate percentage overlap with respect to each curve
    #percentage_overlap_y1 = (overlapping_area / area_y1) * 100
    #percentage_overlap_y2 = (overlapping_area / area_y2) * 100
    #print(f"diff_area {diff_area} area_y1 {area_y1} area_y2 {area_y2}")
    props = {
                "total_rect_control":total_rect_control,
                "diff_area":diff_area,
                "overlapping_area":overlapping_area,
                "area_control":area_control,
                "area_y2":area_y2,
                "diff_overlap_ratio":diff_area/overlapping_area,
                "percent_control_area":area_control/total_rect_control,
            }
    return props


def is_pure_tone(audio_data, sample_rate):
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


    # Find the dominant frequency
    dominant_freq_idx = np.argmax(positive_magnitude)
    dominant_magnitude = positive_magnitude[dominant_freq_idx]

    #print(f"positive_magnitude: {positive_magnitude}")

    positive_magnitude_normalized = positive_magnitude / dominant_magnitude

    #
    # graph_dir = f"./tmp/graph/pure_tone"
    # os.makedirs(graph_dir, exist_ok=True)
    #
    # plt.plot(positive_freqs, positive_magnitude_normalized)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('Frequency Spectrum')
    # #plt.show()
    # plt.savefig(
    #     f'{graph_dir}/{graph_file_name}.png')
    # plt.close()

    # Define a threshold for pure tone
    #noise_threshold = 0.1
    peaks,peak_props = find_peaks(positive_magnitude_normalized, prominence=0.05)  # Adjust prominence

    peak_freqs = positive_freqs[peaks]
    peak_magnitudes = positive_magnitude_normalized[peaks]
    dominant_freq = positive_freqs[dominant_freq_idx]
    # print(f"peak_freqs: {peak_freqs}")
    # print(f"peak_props: {peak_props}")
    # print(f"peak_magnitudes: {peak_magnitudes}")
    # print(f"dominant_freq: {dominant_freq}")
    return len(peaks) == 1 and math.isclose(peak_freqs[0],dominant_freq,rel_tol=0.01)


def max_distance(sorted_data):
    max_dist = 0
    for i in range(1, len(sorted_data)):
        dist = sorted_data[i] - sorted_data[i - 1]
        max_dist = max(max_dist, dist)
    return max_dist
