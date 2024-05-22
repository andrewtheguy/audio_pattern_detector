import unittest
import numpy as np

from process_timestamps import pad_from_backup_intro_ts
from utils import minutes_to_seconds


class TestBackupIntros(unittest.TestCase):

    def do_test(self,intros,backup_intro_ts,news_reports):
        return pad_from_backup_intro_ts(intros,backup_intro_ts,news_reports)
    
    def test_zero_everything(self):
        result = self.do_test(intros=[],
                              backup_intro_ts=[],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [])
        
    def test_no_news(self):
        result = self.do_test(intros=[1,2,3],
                              backup_intro_ts=[],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [1,2,3])
        
    def test_no_backup_intros(self):
        result = self.do_test(intros=[1,2,3],
                              backup_intro_ts=[],
                              news_reports=[4,5,6])
        np.testing.assert_array_equal(result,
                                      [1,2,3])
        
    def test_no_intros(self):
        result = self.do_test(intros=[],
                              backup_intro_ts=[1],
                              news_reports=[4,5,6])
        np.testing.assert_array_equal(result,
                                      [1])
        
    def test_no_intros2(self):
        result = self.do_test(intros=[],
                              backup_intro_ts=[9,1],
                              news_reports=[4,5,6])
        np.testing.assert_array_equal(result,
                                      [1])
        
    def test_equal_size(self):
        result = self.do_test(intros=[1,2,3],
                              backup_intro_ts=[999],
                              news_reports=[4,5,6])
        np.testing.assert_array_equal(result,
                                      [1,2,3])
        
    def test_larger_size(self):
        result = self.do_test(intros=[1,2,3,4],
                              backup_intro_ts=[999],
                              news_reports=[10,11,12])
        np.testing.assert_array_equal(result,
                                      [1,2,3,4])
        
    def test_same_size_consecutive(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)],
                              backup_intro_ts=[minutes_to_seconds(125)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)])
        
    def test_not_making_up_late_intro(self):
        result = self.do_test(intros=[minutes_to_seconds(15),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)],
                              backup_intro_ts=[minutes_to_seconds(5)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(15),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)])
        

    def test_same_size_non_consecutive(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)],
                              backup_intro_ts=[minutes_to_seconds(110)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(50),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)])

    def test_smaller_size_consecutive(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90)],
                              backup_intro_ts=[minutes_to_seconds(120)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(90)])
        
    def test_smaller_size_non_consecutive(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(120)],
                              backup_intro_ts=[minutes_to_seconds(75),minutes_to_seconds(76)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(75),minutes_to_seconds(120)])

    def test_smaller_size_beginning_non_consecutive(self):
        result = self.do_test(intros=[minutes_to_seconds(8),minutes_to_seconds(80),minutes_to_seconds(120)],
                              backup_intro_ts=[minutes_to_seconds(1),minutes_to_seconds(2)],
                              news_reports=[minutes_to_seconds(7),minutes_to_seconds(70),minutes_to_seconds(90),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(1),minutes_to_seconds(8),minutes_to_seconds(80),minutes_to_seconds(120)])


    def test_smaller_size_beginning_no_absorb(self):
        result = self.do_test(intros=[minutes_to_seconds(40),minutes_to_seconds(80),minutes_to_seconds(120)],
                              backup_intro_ts=[minutes_to_seconds(5),minutes_to_seconds(9)],
                              news_reports=[minutes_to_seconds(30),minutes_to_seconds(70),minutes_to_seconds(90),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(5),minutes_to_seconds(40),minutes_to_seconds(80),minutes_to_seconds(120)])

    def test_smaller_size_non_consecutive_too_far(self):
        #with self.assertRaises(ValueError) as cm:
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(120)],
                            backup_intro_ts=[minutes_to_seconds(90)],
                            news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(120)])
        #the_exception = cm.exception
        #self.assertIn("is farther than",str(the_exception))

    def test_smaller_size_matching_two(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(150)],
                              backup_intro_ts=[minutes_to_seconds(75),minutes_to_seconds(76),minutes_to_seconds(110)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130),minutes_to_seconds(160)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(75),minutes_to_seconds(110),minutes_to_seconds(150)])
        
    def test_smaller_size_matching_two_spaced_out(self):
        result = self.do_test(intros=[minutes_to_seconds(30),minutes_to_seconds(60),minutes_to_seconds(150)],
                              backup_intro_ts=[minutes_to_seconds(75),minutes_to_seconds(76),minutes_to_seconds(45)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(50),minutes_to_seconds(70),minutes_to_seconds(130),minutes_to_seconds(160)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30),minutes_to_seconds(45),minutes_to_seconds(60),minutes_to_seconds(75),minutes_to_seconds(150)])
        
if __name__ == '__main__':
    unittest.main()