import json
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

from scrape import logger


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


def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)
