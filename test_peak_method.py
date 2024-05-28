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


def find_closest_troughs(peak_index, data, prominence_threshold=None):
  """Finds the indices of the closest significant troughs to a given peak.

  Args:
    peak_index: The index of the peak in the data series.
    data: The data series as a NumPy array.
    prominence_threshold: Optional. Minimum prominence required for a peak
                          to be considered in trough search. Helps ignore
                          small peaks.

  Returns:
    A tuple containing the indices of the left and right troughs.
  """
  n = len(data)
  left_trough = peak_index
  right_trough = peak_index

  # Search for the left trough
  for i in range(peak_index - 1, -1, -1):
    if data[i] < data[i + 1] and data[i] < data[i - 1]:
      # Check prominence if threshold is provided
      if prominence_threshold is not None:
        if calculate_peak_prominence(i, data) >= prominence_threshold:
          left_trough = i
          break
      else:
        left_trough = i
        break

  # Search for the right trough
  for i in range(peak_index + 1, n):
    if data[i] < data[i + 1] and data[i] < data[i - 1]:
      # Check prominence if threshold is provided
      if prominence_threshold is not None:
        if calculate_peak_prominence(i, data) >= prominence_threshold:
          right_trough = i
          break
      else:
        right_trough = i
        break

  return left_trough, right_trough


def calculate_peak_prominence(peak_index, data):
  """Calculates the prominence of a peak in a data series.

  Args:
    peak_index: The index of the peak in the data series.
    data: The data series as a NumPy array.

  Returns:
    The prominence of the peak.
  """
  left_trough, right_trough = find_closest_troughs(peak_index, data)
  trough_height = max(data[left_trough], data[right_trough])
  prominence = data[peak_index] - trough_height
  return prominence
