import logging


def smooth_preserve_peaks(data, window_size, threshold=0.1):
  """
  Smooths data while preserving sharp peaks.

  Args:
    data: The input data as a 1D numpy array.
    window_size: The size of the smoothing window.
    threshold: The threshold for peak detection (between 0 and 1). Higher values
               result in more peaks preserved.

  Returns:
    The smoothed data as a 1D numpy array.
  """

  # Calculate the rolling average
  smoothed_data = np.convolve(data, np.ones(window_size), 'same') / window_size

  # Find potential peaks
  peaks = np.where(np.diff(np.sign(np.diff(data))) < 0)[0] + 1

  # Preserve peaks based on threshold
  for peak in peaks:
    peak_value = data[peak]
    # Calculate the difference between the peak and the smoothed value
    diff = peak_value - smoothed_data[peak]
    # If the difference is above the threshold, preserve the peak
    if diff > threshold * peak_value:
      smoothed_data[peak - window_size // 2 : peak + window_size // 2] = peak_value

  return smoothed_data

def smooth_preserve_peaks_dist(data, window_size, threshold=0.1, peak_distance=3):
  """
  Smooths data while preserving sharp peaks and removing small peaks.

  Args:
    data: The input data as a 1D numpy array.
    window_size: The size of the smoothing window.
    threshold: The threshold for peak detection (between 0 and 1). Higher values
               result in more peaks preserved.
    peak_distance: Minimum distance between peaks to consider them separate.

  Returns:
    The smoothed data as a 1D numpy array.
  """

  # Calculate the rolling average
  smoothed_data = np.convolve(data, np.ones(window_size), 'same') / window_size

  # Find potential peaks
  peaks = np.where(np.diff(np.sign(np.diff(data))) < 0)[0] + 1

  # Remove small peaks based on distance and threshold
  valid_peaks = []
  for i, peak in enumerate(peaks):
    # Check if this peak is close to another peak
    if i > 0 and abs(peak - peaks[i-1]) < peak_distance:
      continue

    peak_value = data[peak]
    # Calculate the difference between the peak and the smoothed value
    diff = peak_value - smoothed_data[peak]
    # If the difference is above the threshold, preserve the peak
    if diff > threshold * peak_value:
      valid_peaks.append(peak)

  # Preserve valid peaks
  for peak in valid_peaks:
    smoothed_data[peak - window_size // 2 : peak + window_size // 2] = data[peak]

  return smoothed_data

