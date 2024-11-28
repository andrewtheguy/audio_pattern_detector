import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque

import ffmpeg
import numpy as np
import requests
from scipy.integrate import simpson

logger = logging.getLogger(__name__)

# return a tuple of prefix and date
# happydaily20220430 will return ("happydaily","20220430")
# test220220430 will return ("test2","20220430")
def extract_prefix(text):
  match = re.match(r"(.*\d{8,})", text)
  return (match.group(1)[:-8],match.group(1)[-8:]) if match else (None,None)

def get_ffprobe_info(filename):
    """Runs ffprobe command and returns output and return code."""

    cmd = ["ffprobe", "-i", filename, "-v", "quiet", "-print_format", "json", 
           "-show_format", "-show_streams"]

    #try:
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        #returncode = 0  # Success
    #except subprocess.CalledProcessError as e:
    #    result = e.output
    #    returncode = e.returncode  # Error code

    return json.loads(result.decode("utf-8"))  # Decode bytes to string
  
def is_unique_and_sorted(array):
  for i in range(len(array)-1):
    if array[i] >= array[i+1]:
      return False  # Not sorted or unique
  return True  # Unique and sorted

def minutes_to_seconds(minutes):
    return minutes*60


def hours_to_seconds(hours):
    return minutes_to_seconds(hours*60)

def find_nearest_distance_backwards(array, value):
    array = np.asarray(array)
    arr2=(value - array)
    arr2=arr2[arr2 >= 0]
    if len(arr2)==0:
      return None
    return arr2.min()
    #return arr2[idx]

def find_nearest_distance_forward(array, value):
    array = np.asarray(array)
    arr2=(array - value)
    arr2=arr2[arr2 >= 0]
    if len(arr2)==0:
        return None
    return arr2.min()


def get_diff_ratio(control, value):
    diff = np.abs(control - value)
    # max_value = max(input1,input2)
    ratio = diff / control
    return ratio


def slicing_with_zero_padding(array,width,middle_index):
    padding = width/2

    beg = int(middle_index-math.floor(padding))
    end = int(middle_index+math.ceil(padding))

    if beg < 0:
        end = end - beg
        #middle_index = middle_index - beg
        array = np.pad(array, (-beg, 0), 'constant')
        beg = beg - beg


    if end > len(array):
        array = np.pad(array, (0, end - len(array)), 'constant')
    # slice
    return np.array(array[beg:end])

def list_get(my_list, index, default):
    try:
        return my_list[index]
    except IndexError:
        return default


def downsample(values,factor):
    buffer_ = deque([], maxlen=factor)
    downsampled_values = []
    for i, value in enumerate(values):
        buffer_.appendleft(value)
        if (i - 1) % factor == 0:
            # Take max value out of buffer
            # or you can take higher value if their difference is too big, otherwise just average
            max_value = max(buffer_)
            #if max_value > 0.2:
            downsampled_values.append(max_value)
            #else:
            #downsampled_values.append(np.mean(buffer_))
    return np.array(downsampled_values)

def max_distance(sorted_data):
    max_dist = 0
    for i in range(1, len(sorted_data)):
        dist = sorted_data[i] - sorted_data[i - 1]
        max_dist = max(max_dist, dist)
    return max_dist


def calculate_similarity(arr1, arr2):
  """Calculates the similarity between two normalized arrays
     using mean squared error.

  Args:
    arr1: The first normalized array.
    arr2: The second normalized array.

  Returns:
    A similarity score (lower is more similar) based on
    mean squared error.
  """
  return np.mean((arr1 - arr2)**2)


def url_ok(url):


    r = requests.get(url, stream=True)

    if r.ok:
        #content = next(r.iter_content(10))
        return True
    else:
        logger.error(f"HTTP Error {r.status_code} - {r.reason}")
        return False


def download(url,target_file):
    if(os.path.exists(target_file)):
        logger.info(f"file {target_file} already exists,skipping")
        return
    print(f'downloading {url}')
    clip_length_second_stream = float(get_ffprobe_info(url)['format']['duration'])
    with tempfile.TemporaryDirectory() as tmpdir:
        basename,extension = os.path.splitext(os.path.basename(target_file))
        tmp_file = os.path.join(tmpdir,f"download{extension}")
        (
        ffmpeg.input(url).output(tmp_file, **{'bsf:a': 'aac_adtstoasc'}, c='copy', loglevel="error")
              .run()
        )
        clip_length_second_file = float(get_ffprobe_info(tmp_file)['format']['duration'])
        second_tolerate=0.5
        if abs(clip_length_second_file - clip_length_second_stream) > second_tolerate:
            raise ValueError(f"downloaded file duration {clip_length_second_file} does not match stream duration {clip_length_second_stream} by {second_tolerate} seconds")
        shutil.move(tmp_file,target_file)
    print(f'downloaded to {target_file}')

def trapezoidal_area(y1, y2):
    """
    Compute the area between two curves using the trapezoidal rule.

    Args:
    y1 (array): y-coordinates for the first line
    y2 (array): y-coordinates for the second line

    Returns:
    float: The area between the two curves.
    """
    n = len(y1)
    area = 0.0
    for i in range(n - 1):
        # Calculate the width of each section, which is always 1
        h = 1
        # Calculate the average height of the trapezoid
        avg_height = (y1[i] + y1[i + 1] + y2[i] + y2[i + 1]) / 2
        # Calculate the area of the trapezoid
        area += h * avg_height
    return area

def area_of_overlap_ratio(control, variable):
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
    diff_area = area_control+area_y2-2*overlapping_area

    total_area_control = len(control) * max(control)

    # Calculate percentage overlap with respect to each curve
    #percentage_overlap_y1 = (overlapping_area / area_y1) * 100
    #percentage_overlap_y2 = (overlapping_area / area_y2) * 100
    #print(f"diff_area {diff_area} area_y1 {area_y1} area_y2 {area_y2}")
    props = {
                "total_area_control":total_area_control,
                "diff_area":diff_area,
                "overlapping_area":overlapping_area,
                "area_control":area_control,
                "area_y2":area_y2,
                #"diff_area_ratio":diff_area/total_area_control,
                #"overlapping_area_ratio":overlapping_area/total_area_control,
                "percent_control_area":area_control/total_area_control,
            }
    return diff_area/overlapping_area, props