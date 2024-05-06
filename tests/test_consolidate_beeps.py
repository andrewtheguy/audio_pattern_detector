import unittest
import numpy as np

from process_timestamps import consolidate_beeps

class TestConsolidateBeeps(unittest.TestCase):
    def do_test(self,news_reports):
        return consolidate_beeps(list(dict.fromkeys(news_reports)))
    
    def test_zero_everything(self):
        result = self.do_test(news_reports=[])
        np.testing.assert_array_equal(result,[])
        
    def test_only_one(self):
        result = self.do_test(news_reports=[10])
        np.testing.assert_array_equal(result,[10])

    def test_no_repeat(self):
        result = self.do_test(news_reports=[10,30,50,70])
        np.testing.assert_array_equal(result,[10,30,50,70])
        
    def test_repeat_beginning(self):
        result = self.do_test(news_reports=[10,11,12,13, 30,50,70])
        np.testing.assert_array_equal(result,[10,30,50,70])
        
    def test_repeat_middle(self):
        result = self.do_test(news_reports=[10, 30,31,32,33, 50,70])
        np.testing.assert_array_equal(result,[10,30,50,70])

    def test_repeat_end(self):
        result = self.do_test(news_reports=[10,30,50, 70,71,72,73,74])
        np.testing.assert_array_equal(result,[10,30,50,70])
        
    def test_consecutive_good_ones_with_repeat(self):
        result = self.do_test(news_reports=[10,30, 50,51,52,53, 70,71,72,73,74])
        np.testing.assert_array_equal(result,[10,30,50,70])
        
    def test_3_seconds(self):
        result = self.do_test(news_reports=[11,12,13,14, 30,50,70])
        np.testing.assert_array_equal(result,[11,30,50,70])

    def test_4_seconds(self):
        result = self.do_test(news_reports=[11,12,13,14,15, 30,50,70])
        np.testing.assert_array_equal(result,[11,30,50,70])
        
    def test_4_seconds_end(self):
        result = self.do_test(news_reports=[11,30,50,70, 71,72,73,74])
        np.testing.assert_array_equal(result,[11,30,50,70])

    def test_5_seconds(self):
        result = self.do_test(news_reports=[11,12,13,14,15,16, 30,50,70])
        np.testing.assert_array_equal(result,[11,30,50,70])
        
    def test_5_seconds_middle(self):
        result = self.do_test(news_reports=[7, 11,12,13,14,15,16, 30,50,70])
        np.testing.assert_array_equal(result,[7,11,30,50,70])
        
    def test_5_seconds_end(self):
        result = self.do_test(news_reports=[11,30,50, 70,71,72,73,74,75])
        np.testing.assert_array_equal(result,[11,30,50,70])
        
    def test_6_seconds(self):
        result = self.do_test(news_reports=[11,12,13,14,15,16,17, 30,50,70])
        np.testing.assert_array_equal(result,[11,30,50,70])
        
    def test_6_seconds_middle(self):
        result = self.do_test(news_reports=[7, 11,12,13,14,15,16,17, 30,50,70])
        np.testing.assert_array_equal(result,[7,11,30,50,70])
        
    def test_6_seconds_beginning_middle_and_end(self):
        result = self.do_test(news_reports=[0,1,2,3,4,5,6, 11,30,50, 70,71,72,73,74,75,76, 80, 90,91,92])
        np.testing.assert_array_equal(result,[0,11,30,50,70,80,90])
        
    def test_2_seconds_between(self):
        result = self.do_test(news_reports=[0,1,2,4,5,6, 11,30,50, 70,71,72,74,75,76, 80, 90,91,92])
        np.testing.assert_array_equal(result,[0,11,30,50,70,80,90])
        
    def test_2_seconds_between_repeat_6(self):
        result = self.do_test(news_reports=[0,1,2,4,5,6,7, 11,30,50, 70,71,72,74,75,76, 80, 90,91,92])
        np.testing.assert_array_equal(result,[0,7,11,30,50,70,80,90])
        
    def test_6_seconds_end(self):
        result = self.do_test(news_reports=[11,30,50, 70,71,72,73,74,75,76])
        np.testing.assert_array_equal(result,[11,30,50,70])
        

if __name__ == '__main__':
    unittest.main()