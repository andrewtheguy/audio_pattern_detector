import unittest
import numpy as np

from process_timestamps import INTRO_CUT_OFF, pad_news_report
from utils import minutes_to_seconds

class TestPadNewsReport(unittest.TestCase):

    def do_test(self,time_sequences):
        return pad_news_report(time_sequences)

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
        
        
if __name__ == '__main__':
    unittest.main()