import unittest
import numpy as np

from process_timestamps import INTRO_CUT_OFF, pad_news_report
from utils import minutes_to_seconds

class TestPadNewsReport(unittest.TestCase):
    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)

    def do_test(self,time_sequences):
        return pad_news_report(time_sequences,total_time=self.total_time_1,news_report_second_pad=6)

    def test_pad_zero(self):
        result = self.do_test(time_sequences=[
                                            ])
        np.testing.assert_array_equal(result,
                                       [
                                       ])
        
    def test_pad(self):

        result = self.do_test(time_sequences=[[30,60],
                                              [90,120],
                                              [140,160],
                                              [180,200],
                                            ])
        np.testing.assert_array_equal(result,
                                       [[30,66],
                                        [90,126],
                                        [140,166],
                                        [180,206],
                                    ])
        
    def test_pad_total(self):

        result = self.do_test(time_sequences=[[30,60],
                                              [90,120],
                                              [140,160],
                                              [180,self.total_time_1],
                                            ])
        np.testing.assert_array_equal(result,
                                       [[30,66],
                                        [90,126],
                                        [140,166],
                                        [180,self.total_time_1],
                                    ])
        
    def test_pad2_total_2(self):

        result = self.do_test(time_sequences=[[30,60],
                                              [90,120],
                                              [140,160],
                                              [180,self.total_time_1-2],
                                            ])
        np.testing.assert_array_equal(result,
                                       [[30,66],
                                        [90,126],
                                        [140,166],
                                        [180,self.total_time_1],
                                    ])
        
    def test_pad_overlap(self):

        result = self.do_test(time_sequences=[[30,60],
                                              [60,120],
                                              [122,160],
                                              [162,206],
                                            ])
        np.testing.assert_array_equal(result,
                                       [[30,60],
                                        [60,122],
                                        [122,162],
                                        [162,212],
                                    ])
        
        
if __name__ == '__main__':
    unittest.main()