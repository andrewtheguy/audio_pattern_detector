import unittest
import numpy as np

from process_timestamps import fill_in_short_intervals_missing_intros, pad_from_backup_intro_ts
from utils import minutes_to_seconds


class TestFillMissingShortGaps(unittest.TestCase):

    def do_test(self,intros,news_reports):
        return fill_in_short_intervals_missing_intros(intros,news_reports)
    
    def test_zero_everything(self):
        result = self.do_test(intros=[],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [])
        
    def test_no_news(self):
        result = self.do_test(intros=[1,2,3],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [1,2,3])
        
    def test_no_intros(self):
        result = self.do_test(intros=[],
                              news_reports=[4,5,6])
        np.testing.assert_array_equal(result,
                                      [])
        
        
    def test_equal_size(self):
        result = self.do_test(intros=[1,2,3],
                              news_reports=[4,5,6])
        np.testing.assert_array_equal(result,
                                      [1,2,3])
        
    def test_larger_size(self):
        result = self.do_test(intros=[1,2,3,4],
                              news_reports=[10,11,12])
        np.testing.assert_array_equal(result,
                                      [1,2,3,4])
        
    def test_same_size_consecutive(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)])

    def test_making_up_short_duration(self):
        result = self.do_test(intros=[minutes_to_seconds(15),minutes_to_seconds(90),minutes_to_seconds(120)],
                              
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(60),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(15),minutes_to_seconds(40),minutes_to_seconds(90),minutes_to_seconds(120)])
        

    def test_not_making_up_long_duration(self):
        result = self.do_test(intros=[minutes_to_seconds(15),minutes_to_seconds(90),minutes_to_seconds(120)],
                              
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(15),minutes_to_seconds(90),minutes_to_seconds(120)])
        

    def test_same_size_non_consecutive(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(50),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)])

    def test_smaller_size_consecutive(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90)])
        
    def test_smaller_size_non_consecutive(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(120)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(120)])

    def test_not_make_up_beginning_non_consecutive_long_duration(self):
        result = self.do_test(intros=[minutes_to_seconds(80),minutes_to_seconds(120)],
                              news_reports=[minutes_to_seconds(30),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(80),minutes_to_seconds(120)])
        
    def test_make_up_beginning_non_consecutive_short_duration(self):
        result = self.do_test(intros=[minutes_to_seconds(80),minutes_to_seconds(120)],
                              news_reports=[minutes_to_seconds(20),minutes_to_seconds(90),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [0,minutes_to_seconds(80),minutes_to_seconds(120)])

    def test_smaller_size_matching_two(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(150)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(90),minutes_to_seconds(110),minutes_to_seconds(160)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(70),minutes_to_seconds(90),minutes_to_seconds(150)])
        
    def test_smaller_size_matching_two_spaced_out(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(150)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(50),minutes_to_seconds(110),minutes_to_seconds(130),minutes_to_seconds(160)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(40),minutes_to_seconds(60),minutes_to_seconds(110),minutes_to_seconds(150)])
        
    def test_not_make_up_end_long(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(80),minutes_to_seconds(110)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60)])

    def test_make_up_end(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(80),minutes_to_seconds(100)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(80)])

if __name__ == '__main__':
    unittest.main()