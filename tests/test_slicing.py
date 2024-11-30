import numpy as np

from audio_pattern_detector.audio_utils import slicing_with_zero_padding


def test_slice_odd():
    arr = [1, 2, 3, 4, 5]
    width = 3
    middle_index = 2
    np.testing.assert_array_equal(slicing_with_zero_padding(arr, width, middle_index),[2, 3, 4])

def test_slice_even():
    arr = [1, 2, 3, 4, 5]
    width = 4
    middle_index = 2
    np.testing.assert_array_equal(slicing_with_zero_padding(arr, width, middle_index),[1, 2, 3, 4])

def test_slice_end_short():
    arr = [1, 2, 3, 4, 5]
    width = 4
    middle_index = 4
    arr_result = slicing_with_zero_padding(arr, width, middle_index)
    #print(arr_result)
    np.testing.assert_array_equal(arr_result,[3, 4, 5, 0])

def test_slice_end_short_odd():
    arr = [1, 2, 3, 4, 5]
    width = 5
    middle_index = 3
    arr_result = slicing_with_zero_padding(arr, width, middle_index)
    #print(arr_result)
    np.testing.assert_array_equal(arr_result,[2, 3, 4, 5, 0])

def test_slice_beg_short():
    arr = [1, 2, 3, 4, 5]
    width = 4
    middle_index = 1
    np.testing.assert_array_equal(slicing_with_zero_padding(arr, width, middle_index),[0, 1, 2, 3])

def test_slice_beg_short_odd():
    arr = [1, 2, 3, 4, 5]
    width = 5
    middle_index = 1
    np.testing.assert_array_equal(slicing_with_zero_padding(arr, width, middle_index),[0, 1, 2, 3, 4])



