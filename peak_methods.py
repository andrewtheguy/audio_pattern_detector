import logging

import numpy as np
from scipy.signal import peak_widths


def find_closest_troughs(peak_index, data):
  """Finds the indices of the closest significant troughs to a given peak.

  Args:
    peak_index: The index of the peak in the data series.
    data: The data series as a NumPy array.


  Returns:
    A tuple containing the indices of the left and right troughs.
  """
  n = len(data)
  left_trough = peak_index
  right_trough = peak_index

  # not a peak index
  if peak_index == 0 or peak_index == n - 1:
    logging.warning("Peak index is at the edge of the data series.")
    return left_trough, right_trough

  # Search for the left trough
  for i in range(peak_index - 1, -1, -1):
    # plateaus
    if data[i] == data[i+1]:
        left_trough = i
        continue
    if data[i] < data[i + 1] and i-1 >= 0 and data[i] < data[i - 1]:
        left_trough = i
        break
  if left_trough == 1 and data[0] < data[1]:
    left_trough = 0
  elif left_trough == peak_index and data[0] < data[peak_index]:
    left_trough = 0

  # Search for the right trough
  for i in range(peak_index + 1, n):
    # plateaus
    if data[i] == data[i-1]:
        right_trough = i
        continue
    if i+1 < n and data[i] < data[i + 1] and data[i] < data[i - 1]:
        right_trough = i
        break
  if right_trough == n - 2 and data[n-1] < data[n-2]:
    right_trough = n - 1
  elif right_trough == peak_index and data[n-1] < data[peak_index]:
    right_trough = n - 1

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
  return prominence,left_trough, right_trough

def get_peak_profile(peak_index, data):
  prominence, left_trough, right_trough = calculate_peak_prominence(peak_index, data)
  prominences = (np.array([prominence], dtype="float64"), np.array([left_trough]), np.array([right_trough]))
  #print("prominences", prominences)
  width_100 =  right_trough - left_trough
  if width_100 == 0:
    logging.warning(f"Peak index {peak_index} is at the edge of the data series.")
    width_75 = 0
    width_50 = 0
  else:
    width_75 = peak_widths(data, [peak_index], rel_height=0.75, prominence_data=prominences)[0]
    width_50 = peak_widths(data, [peak_index], rel_height=0.5, prominence_data=prominences)[0]
  return {"width_100": width_100, "width_75": width_75, "width_50": width_50,"prominence":prominence,
          "left_trough":left_trough,"right_trough":right_trough}