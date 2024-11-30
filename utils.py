import json
import logging
import os
import re
import shutil
import subprocess
import tempfile

import ffmpeg
import numpy as np
import requests

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


def list_get(my_list, index, default):
    try:
        return my_list[index]
    except IndexError:
        return default

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

