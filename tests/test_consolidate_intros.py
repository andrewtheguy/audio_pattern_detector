import unittest
import numpy as np

from process_timestamps import consolidate_intros
from time_sequence_error import TimeSequenceError
from utils import minutes_to_seconds

class TestConsolidateIntros(unittest.TestCase):
    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)
        
    def do_test(self,intros,news_reports,backup_intro_ts=[]):
        return consolidate_intros((list(dict.fromkeys(intros))),(list(dict.fromkeys(news_reports))),self.total_time_1,backup_intro_ts=backup_intro_ts)
    
    def test_intro_negative(self):
        with self.assertRaises(ValueError) as cm:
            result = self.do_test(intros=      [-1,800],
                                news_reports=[300,1200,5000])
        the_exception = cm.exception
        self.assertIn("less than 0",str(the_exception))
        
    def test_intro_more_than_total(self):
        with self.assertRaises(TimeSequenceError) as cm:
            result = self.do_test(intros=      [100,self.total_time_1+10],
                                news_reports=[300])
        the_exception = cm.exception
        self.assertIn("overflow",str(the_exception))
        

    def test_zero_everything(self):
        result = self.do_test(intros=[],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [0])
        
    def test_news_reports_only(self):
        result = self.do_test(intros=[],
                              news_reports=[3,4])
        np.testing.assert_array_equal(result,
                                      [0])
        result = self.do_test(intros=[],
                              news_reports=[3,50])
        np.testing.assert_array_equal(result,
                                      [0])
        
    def test_intros_only(self):
        result = self.do_test(intros=[3],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [3])
        
        # if no news reports, but multiple intros, only return first one
        result = self.do_test(intros=[3,4],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [3])
        
    def test_news_report_in_between_intros(self):
        result = self.do_test(intros=[20,70],
                              news_reports=[50])
        np.testing.assert_array_equal(result,
                                      [20,70])
        
        result = self.do_test(intros=[20,50],
                              news_reports=[50])
        np.testing.assert_array_equal(result,
                                      [20,50])
        
        result = self.do_test(intros=   [20,50,70],
                              news_reports=[50])
        np.testing.assert_array_equal(result,
                                      [20,50])
        
    def test_no_extras_only(self):
        result = self.do_test(intros=      [100,800],
                              news_reports=[300,1200,5000])
        np.testing.assert_array_equal(result,
                                      [100,800])
        
        result = self.do_test(intros=      [300,800,1800],
                              news_reports=[400,1500,2800])
        np.testing.assert_array_equal(result,
                                      [300,800,1800])
        
        result = self.do_test(intros=      [300,800,1800,3200],
                              news_reports=[400,1500,2800])
        np.testing.assert_array_equal(result,
                                      [300,800,1800,3200])
        
        result = self.do_test(intros=      [400],
                              news_reports=[800])
        np.testing.assert_array_equal(result,
                                      [400])
        
        result = self.do_test(intros=      [800],
                              news_reports=[400])
        np.testing.assert_array_equal(result,
                                      [800])
        
    def test_extras_news_reports_only(self):      
        result = self.do_test(intros=      [800],
                              news_reports=[400,1500,2800])
        np.testing.assert_array_equal(result,
                                      [800])
        
        result = self.do_test(intros=          [1600,2900],
                              news_reports=[400,1500,2800,3200,3400])
        np.testing.assert_array_equal(result,
                                      [1600,2900])
        
        #result = self.do_test(intros=               [1600,     3300],
        #                      news_reports=[400,1500,2800,3200,3400,3500])
        #np.testing.assert_array_equal(result,
        #                              [1600,3300])
        

        
    def test_extras_no_overlapping(self):
        result = self.do_test(intros=      [100,200,800,810,820],
                              news_reports=[    300,        1200,5000])
        np.testing.assert_array_equal(result,
                                      [100,800])
        
        result = self.do_test(intros=      [300,800,1800,     2801,2802],
                              news_reports=[400,1500,    2800])
        np.testing.assert_array_equal(result,
                                      [300,800,1800,2801])
        
    def test_extras_no_overlapping_repeating(self):
        result = self.do_test(intros=      [100,200,200,800,810,810,820],
                              news_reports=[        300,            1200,5000])
        np.testing.assert_array_equal(result,
                                      [100,800])
        
        
    def test_extras_overlapping(self):
        result = self.do_test(intros=      [100,300,800,810,1200],
                              news_reports=[300,       1200,5000])
        np.testing.assert_array_equal(result,
                                      [100,300,1200])
        
        result = self.do_test(intros=      [300,800,1800,     2800],
                              news_reports=[400,1500,    2800])
        np.testing.assert_array_equal(result,
                                      [300,800,1800,2800])
        
        result = self.do_test(intros=      [300,800,1800,     2800,2900],
                              news_reports=[400,1500,    2800])
        np.testing.assert_array_equal(result,
                                      [300,800,1800,2800])
        
    def test_extras_overlapping_repeating(self):
        result = self.do_test(intros=      [100,200,200,300,300,800,810,810,820],
                              news_reports=[                300,            1200,5000])
        np.testing.assert_array_equal(result,
                                      [100,300])
        
    def test_making_up_late_intro(self):
        result = self.do_test(intros=[minutes_to_seconds(15),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)],
                              backup_intro_ts=[minutes_to_seconds(5)],
                              news_reports=[minutes_to_seconds(40),minutes_to_seconds(70),minutes_to_seconds(100),minutes_to_seconds(130)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(5),minutes_to_seconds(60),minutes_to_seconds(90),minutes_to_seconds(120)])

    def test_making_up_missing_intro_end(self):
        result = self.do_test(
            intros=[minutes_to_seconds(30),minutes_to_seconds(60), minutes_to_seconds(90)],
            backup_intro_ts=[minutes_to_seconds(120)],
            news_reports=[minutes_to_seconds(40), minutes_to_seconds(70), minutes_to_seconds(100)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30), minutes_to_seconds(60), minutes_to_seconds(90),minutes_to_seconds(120)])

    def test_not_making_up_missing_intro_end_if_between(self):
        result = self.do_test(
            intros=[minutes_to_seconds(30),minutes_to_seconds(60), minutes_to_seconds(90)],
            backup_intro_ts=[minutes_to_seconds(95)],
            news_reports=[minutes_to_seconds(40), minutes_to_seconds(70), minutes_to_seconds(100)])
        np.testing.assert_array_equal(result,
                                      [minutes_to_seconds(30), minutes_to_seconds(60), minutes_to_seconds(90)])



if __name__ == '__main__':
    unittest.main()