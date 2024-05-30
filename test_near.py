
import numpy as np

from utils import find_nearest_distance_backwards, find_nearest_distance_forward

def test_find_nearest_distance_backwards():
    assert find_nearest_distance_backwards([1,2,3,9,10,11],3.5) == 0.5
    assert find_nearest_distance_backwards([1,2,3,10,7.5,9,10,11],8) == 0.5
    assert find_nearest_distance_backwards([],3.5) == None

    # not sorted or unique
    assert find_nearest_distance_backwards([2, 9, 1, 10, 11, 9, 3], 3.5) == 0.5

def test_find_nearest_distance_forward():
    assert find_nearest_distance_forward([1,2,3,9,10,11],3.5) == 5.5
    assert find_nearest_distance_forward([1,2,3,9,10,11],99) == None
    assert find_nearest_distance_forward([],3.5) == None

    # not sorted or unique
    assert find_nearest_distance_forward([2, 9, 1, 10, 11, 9, 3], 3.5) == 5.5