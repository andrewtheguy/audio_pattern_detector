import unittest
import numpy as np

from process_timestamps import INTRO_CUT_OFF, news_intro_cut_off_beginning_and_end, remove_start_equals_to_end
from utils import minutes_to_seconds

class TestRemoveStartEnd(unittest.TestCase):

    def do_test(self,time_sequences):
        return remove_start_equals_to_end(time_sequences)
    
 
        
    def test_remove(self):

        result = self.do_test(time_sequences=[[30,60],
                                                          [90,90],
                                                          [120,140],
                                                          [180,180],
                                                          ])
        np.testing.assert_array_equal(result,
                                      [[30,60],
                                       [120,140],
                                       ])
        
        
if __name__ == '__main__':
    unittest.main()