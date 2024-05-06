import unittest
import numpy as np

from process_timestamps import consolidate_intros

class TestConsolidateIntros(unittest.TestCase):
    def do_test(self,intros,news_reports):
        return consolidate_intros((list(dict.fromkeys(intros))),(list(dict.fromkeys(news_reports))))
    
    def test_zero_everything(self):
        result = self.do_test(intros=[],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [])
        
    def test_news_reports_only(self):
        result = self.do_test(intros=[],
                              news_reports=[3,4])
        np.testing.assert_array_equal(result,
                                      [])
        result = self.do_test(intros=[],
                              news_reports=[3,4])
        np.testing.assert_array_equal(result,
                                      [])
        
    def test_intros_only(self):
        result = self.do_test(intros=[3],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [3])
        
        result = self.do_test(intros=[3,4],
                              news_reports=[])
        np.testing.assert_array_equal(result,
                                      [3,4])
        
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
        
if __name__ == '__main__':
    unittest.main()