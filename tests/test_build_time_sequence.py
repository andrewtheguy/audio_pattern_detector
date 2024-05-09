import unittest
import numpy as np

from process_timestamps import INTRO_CUT_OFF, build_time_sequence
from time_sequence_error import TimeSequenceError
from utils import minutes_to_seconds

class TestBuildTimeSequence(unittest.TestCase):

    def do_test(self,intros,news_reports):
        return build_time_sequence(intros,news_reports)
    
 
        
    def test_build_valid(self):

        result = self.do_test([300,900,1400],
                              [600,1200,1600],
                            )
        np.testing.assert_array_equal(result,
                                      [[300,600],
                                       [900,1200],
                                       [1400,1600],
                                       ])
        
    def test_skip_start_equals_to_end(self):

        result = self.do_test([300,900, 1300, 1400],
                              [600,1200,1300, 1600],
                            )
        np.testing.assert_array_equal(result,
                                      [[300,600],
                                       [900,1200],
                                       [1400,1600],
                                       ])
        
    def test_build_invalid(self):
        with self.assertRaises(TimeSequenceError):
            result = self.do_test([300,900,1400],
                                [600,1200],
                                )
    
        
    def test_build_invalid2(self):
        with self.assertRaises(TimeSequenceError):
            result = self.do_test([300,900],
                                [600,1200,1600],
                                )
 
        
    def test_build_invalid3(self):
        with self.assertRaises(TimeSequenceError):
            result = self.do_test([],
                                [600,1200,1600],
                                )

        with self.assertRaises(TimeSequenceError):        
            result = self.do_test([600,1200,1600],
                                [],
                                )

        
    def test_build_empty(self):

        result = self.do_test([],
                             []
                              )
        np.testing.assert_array_equal(result,
                                      [
                                       ])
        
if __name__ == '__main__':
    unittest.main()