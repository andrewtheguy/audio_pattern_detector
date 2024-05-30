import json
import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from numpy_encoder import NumpyEncoder
from utils import downsample


def verify_peak(sr,max_index,correlation,audio_section,section_ts,clip_name,index,debug_mode):
    factor = sr/10

    print(max_index)

    #wlen = max(int(sr/2), int(clip_length))
    padding = sr*4

    beg = max(int(max_index-padding), 0)
    end = min(len(audio_section),int(max_index+padding))
    #print("chafa")
    #print(beg,end)
    #exit(1)

    correlation = correlation[beg:end]
    correlation = downsample(correlation,int(factor))

    if debug_mode:
        graph_dir = f"./tmp/graph/resampled/{clip_name}"
        os.makedirs(graph_dir, exist_ok=True)

        #Optional: plot the correlation graph to visualize
        plt.figure(figsize=(10, 4))
        # if clip_name == "漫談法律intro" and index == 10:
        #     plt.plot(correlation[454000:454100])
        # elif clip_name == "漫談法律intro" and index == 11:
        #     plt.plot(correlation[50000:70000])
        # elif clip_name == "日落大道smallinterlude" and index == 13:
        #     plt.plot(correlation[244100:244700])
        # elif clip_name == "日落大道smallinterlude" and index == 14:
        #     plt.plot(correlation[28300:28900])
        # elif clip_name == "繼續有心人intro" and index == 10:
        #     plt.plot(correlation[440900:441000])
        # else:
        #     plt.plot(correlation)
        plt.plot(correlation)

        plt.title('Cross-correlation between the audio clip and full track before slicing')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coefficient')
        plt.savefig(
            f'{graph_dir}/{clip_name}_{index}_{section_ts}.png')
        plt.close()

    peaks, properties = find_peaks(correlation, width=0, threshold=0, wlen=10, height=0, prominence=0.2, rel_height=1)
    if debug_mode:
        peak_dir = f"./tmp/peaks/resampled/{clip_name}"
        os.makedirs(peak_dir, exist_ok=True)
        peaks_test=[]
        for i,item in enumerate(peaks):
            #plot_test_x=np.append(plot_test_x, index)
            #plot_test_y=np.append(plot_test_y, item)
            peaks_test.append([{"index":int(item),"second":item/sr,
                                "height":properties["peak_heights"][i],
                                "prominence":properties["prominences"][i],
                                "width":properties["widths"][i],
                               }])
        peaks_test.append({"properties":properties})
        print(json.dumps(peaks_test, indent=2,cls=NumpyEncoder), file=open(f'{peak_dir}/{clip_name}_{index}_{section_ts}.txt', 'w'))


    if len(peaks) != 1:
        print(f"failed verification for {section_ts} due to multiple peaks {peaks} or zero peaks")
        return False

    index_final=0
    #peak = peaks[index_final]
    passed = properties["peak_heights"][index_final] == 1.0 and properties["prominences"][index_final] > 0.7 and properties["widths"][index_final] <= 10
    if not passed:
        print(f"failed verification for {section_ts} due to peak {peaks[index_final]} not meeting requirements")
    return passed


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

