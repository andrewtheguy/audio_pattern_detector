import unittest
import numpy as np

from process_timestamps import process_timestamps, timestamp_sanity_check
from utils import minutes_to_seconds
from time_sequence_error import TimeSequenceError

class TestDurationAndGaps(unittest.TestCase):
    
    def check(self,result,allow_first_short=False):
        return timestamp_sanity_check(result,skip_reasonable_time_sequence_check=False,allow_first_short=allow_first_short)

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            result = self.check([])

    def test_empty_2d(self):
        with self.assertRaises(ValueError):
            result = self.check([[]])

    def test_valid(self):
        try:
            result = self.check([[1,minutes_to_seconds(18)]])
        except Exception as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")
        
    def test_1_value(self):
        with self.assertRaises(ValueError):
            result = self.check([[1]])

    def test_3_values(self):
        with self.assertRaises(ValueError):
            result = self.check([[1,2,3]])

    def test_valid(self):
        try:
            result = self.check([[1,minutes_to_seconds(18)]])
        except Exception as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")

    def test_too_short(self):
        with self.assertRaises(TimeSequenceError):
            result = self.check([[1,minutes_to_seconds(14)]])

    def test_allow_first_short(self):
        #with self.assertRaises(TimeSequenceError):
        result = self.check([[1,minutes_to_seconds(6)]],allow_first_short=True)
        result = self.check([
                [4,minutes_to_seconds(9)],
                [minutes_to_seconds(12), minutes_to_seconds(30)],
            ],allow_first_short=True)
        
    def test_allow_first_short_only_close_enough_to_beginning(self):
        with self.assertRaises(TimeSequenceError):
            result = self.check([
                [1,minutes_to_seconds(13)],
                [minutes_to_seconds(14), minutes_to_seconds(18)],
            ],
            allow_first_short=True)


    def test_gap_normal(self):
        try:
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(25),minutes_to_seconds(50)]])
        except TimeSequenceError as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")

    def test_allow_continuous(self):
        try:
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(20),minutes_to_seconds(50)]])
        except Exception as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")

    def test_gap_too_large(self):
        with self.assertRaises(TimeSequenceError):
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(50),minutes_to_seconds(60)]])

    def test_disallow_flip_over(self):
        with self.assertRaises(ValueError):
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(19),minutes_to_seconds(60)]])

    def test_disallow_negative(self):
        with self.assertRaises(ValueError):
            result = self.check([[-1,minutes_to_seconds(20)],[minutes_to_seconds(25),minutes_to_seconds(50)]])

    def test_disallow_overlap(self):
        with self.assertRaises(ValueError):
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(19),minutes_to_seconds(50)]])
