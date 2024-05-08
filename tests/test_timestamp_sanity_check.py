import unittest
import numpy as np

from process_timestamps import timestamp_sanity_check
from utils import minutes_to_seconds
from time_sequence_error import TimeSequenceError

class TestTimestampSanityCheck(unittest.TestCase):
    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)

    def check(self,result,allow_first_short=False):
        return timestamp_sanity_check(result,total_time=self.total_time_1)


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

    def test_intro_negative(self):
        with self.assertRaises(ValueError) as cm:
            result = self.check([[-1,minutes_to_seconds(18)]])
        the_exception = cm.exception
        self.assertIn("less than 0",str(the_exception))
    
    def test_intro_overflow(self):
        with self.assertRaises(ValueError) as cm:
            result = self.check([[self.total_time_1+10,self.total_time_1+minutes_to_seconds(18)]])
        the_exception = cm.exception
        self.assertIn("overflow",str(the_exception))
        
    def test_1_value(self):
        with self.assertRaises(ValueError):
            result = self.check([[1]])

    def test_3_values(self):
        with self.assertRaises(ValueError):
            result = self.check([[1,2,3]])

    def test_valid(self):
        try:
            result = self.check([[1,minutes_to_seconds(18)],[minutes_to_seconds(19),minutes_to_seconds(21)]])
        except Exception as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")


    def test_allow_continuous(self):
        try:
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(20),minutes_to_seconds(50)]])
        except Exception as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")

    def test_disallow_out_of_order(self):
        with self.assertRaises(ValueError):
            result = self.check([[minutes_to_seconds(20),minutes_to_seconds(50)],[1,minutes_to_seconds(19)]])

    def test_disallow_negative(self):
        with self.assertRaises(ValueError):
            result = self.check([[-1,minutes_to_seconds(20)],[minutes_to_seconds(25),minutes_to_seconds(50)]])

    def test_disallow_overlap(self):
        with self.assertRaises(ValueError):
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(19),minutes_to_seconds(50)]])
