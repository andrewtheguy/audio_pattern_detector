import json
import math
import re
import subprocess

import numpy as np


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

if __name__ == '__main__':
  array1 = [1, 2, 3, 4, 5]  # Unique and sorted
  array2 = [1, 2, 2, 4, 5]  # Not unique
  array3 = [1, 4, 2, 3, 5]  # Unique but not sorted
  array4 = [1, 2, 4, 2, 5]  # Not unique

  assert(is_unique_and_sorted(array1)==True)  # Output: True
  assert(is_unique_and_sorted(array2)==False)  # Output: False
  assert(is_unique_and_sorted(array3)==False)  # Output: False
  assert(is_unique_and_sorted(array4)==False)  # Output: False

  def list_get(my_list, index, default):
    try:
        return my_list[index]
    except IndexError:
        return default