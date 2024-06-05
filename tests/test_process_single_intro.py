import unittest
import numpy as np

from process_timestamps import process_timestamps_simple
from time_sequence_error import TimeSequenceError
from utils import minutes_to_seconds

class TestProcessSingleIntro(unittest.TestCase):
    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)

    def do_test(self,intros,endings,expected_num_segments=None,ends_with_intro=False,intro_max_repeat_seconds=None):
        return process_timestamps_simple((list(dict.fromkeys(intros))),(list(dict.fromkeys(endings))),self.total_time_1,
                                          expected_num_segments=expected_num_segments,ends_with_intro=ends_with_intro,intro_max_repeat_seconds=intro_max_repeat_seconds)
    
    def test_not_allow_zero_everything(self):
        with self.assertRaises(ValueError) as cm:
            result_news_report = self.do_test(intros=[],
                                endings=[])
            np.testing.assert_array_equal(result_news_report,
                                        [self.total_time_1])
        the_exception = cm.exception
        self.assertIn("intros cannot be empty",str(the_exception))
        
    def test_not_allow_endings_only(self):
        with self.assertRaises(ValueError) as cm:
            self.do_test(intros=[],
                              endings=[minutes_to_seconds(3),minutes_to_seconds(5)])
        the_exception = cm.exception
        self.assertIn("intros cannot be empty",str(the_exception))
        
    def test_ending_overflow(self):
        with self.assertRaises(TimeSequenceError) as cm:
            self.do_test(intros=           [minutes_to_seconds(3),minutes_to_seconds(60)],
                              endings=[4,self.total_time_1+10])
        the_exception = cm.exception
        self.assertIn("ending overflow",str(the_exception))
        
    def test_intro_overflow(self):
        with self.assertRaises(TimeSequenceError) as cm:
            self.do_test(intros=           [3,self.total_time_1+10],
                              endings=[4,self.total_time_1-30])
        the_exception = cm.exception
        self.assertIn("intro overflow",str(the_exception))
        
    def test_intros_only(self):
        result = self.do_test(intros=[3],
                              endings=[])
        np.testing.assert_array_equal(result,
                                      [[3,self.total_time_1]])
        
    def test_not_allow_no_intro(self):
        with self.assertRaises(ValueError) as cm:
            result = self.do_test(intros=      [],
                                endings=[minutes_to_seconds(20),minutes_to_seconds(40)])
        the_exception = cm.exception
        self.assertIn("intros cannot be empty",str(the_exception))

    def test_normal(self):
        result = self.do_test(intros=      [minutes_to_seconds(3),minutes_to_seconds(30)],
                              endings=[minutes_to_seconds(20),minutes_to_seconds(40)])
        np.testing.assert_array_equal(result,
                                      [
                                        [minutes_to_seconds(3),minutes_to_seconds(20)],
                                        [minutes_to_seconds(30),minutes_to_seconds(40)],
                                       ])

    def test_ending_before_intro(self):
        result = self.do_test(intros=      [minutes_to_seconds(3),minutes_to_seconds(30)],
                              endings=[minutes_to_seconds(2),minutes_to_seconds(20),minutes_to_seconds(40)])
        np.testing.assert_array_equal(result,
                                      [
                                        [minutes_to_seconds(3),minutes_to_seconds(20)],
                                        [minutes_to_seconds(30),minutes_to_seconds(40)],
                                       ])

    def test_one_less(self):
        result = self.do_test(intros=      [minutes_to_seconds(3),minutes_to_seconds(30)],
                              endings=[minutes_to_seconds(20)])
        np.testing.assert_array_equal(result,
                                      [
                                        [minutes_to_seconds(3),minutes_to_seconds(20)],
                                        [minutes_to_seconds(30),self.total_time_1],
                                       ])
        
    def test_ends_with_intro_one_intro(self):
        #with self.assertRaises(ValueError) as cm:
            result = self.do_test(intros=      [minutes_to_seconds(3)],
                                endings=[],ends_with_intro=True)
            np.testing.assert_array_equal(result,
                                        [
                                            [minutes_to_seconds(3),self.total_time_1],
                                        ])
        #the_exception = cm.exception
        #self.assertIn("no enough",str(the_exception))


    def test_ends_with_intro(self):
        result = self.do_test(intros= [minutes_to_seconds(3),minutes_to_seconds(30),minutes_to_seconds(40)],
                              endings=[],ends_with_intro=True)
        np.testing.assert_array_equal(result,
                                      [
                                        [minutes_to_seconds(3),minutes_to_seconds(30)],
                                        [minutes_to_seconds(30),minutes_to_seconds(40)],
                                       ])
        
    def test_ends_with_intro_with_ending_greater_than_intro(self):
        result = self.do_test(intros=      [minutes_to_seconds(3),minutes_to_seconds(30)],
                            endings=[minutes_to_seconds(20),minutes_to_seconds(60)],ends_with_intro=True)
        np.testing.assert_array_equal(result,
                                    [
                                        [minutes_to_seconds(3),minutes_to_seconds(20)],
                                        [minutes_to_seconds(30),minutes_to_seconds(60)],
                                    ])

    def test_ends_with_intro_with_another_ending_between(self):
        result = self.do_test(intros=      [minutes_to_seconds(3),minutes_to_seconds(30)],
                            endings=[minutes_to_seconds(20)],ends_with_intro=True)
        np.testing.assert_array_equal(result,
                                    [
                                        [minutes_to_seconds(3),minutes_to_seconds(20)],
                                    ])
        
    def test_ends_with_intro_with_another_ending_between2(self):

        result = self.do_test(intros=      [minutes_to_seconds(3),minutes_to_seconds(30),minutes_to_seconds(60)],
                            endings=[minutes_to_seconds(20)],ends_with_intro=True)
        np.testing.assert_array_equal(result,
                                    [
                                        [minutes_to_seconds(3),minutes_to_seconds(20)],
                                        [minutes_to_seconds(30),minutes_to_seconds(60)],
                                    ])
        
    def test_segment_count(self):
        result = self.do_test(intros=      [minutes_to_seconds(3),minutes_to_seconds(30)],
                              endings=[minutes_to_seconds(20),minutes_to_seconds(40)],expected_num_segments=2)
        np.testing.assert_array_equal(result,
                                      [
                                        [minutes_to_seconds(3),minutes_to_seconds(20)],
                                        [minutes_to_seconds(30),minutes_to_seconds(40)],
                                       ])

    def test_unexpected_segment_count(self):
        with self.assertRaises(TimeSequenceError) as cm:
            result = self.do_test(intros=      [minutes_to_seconds(3),minutes_to_seconds(30)],
                                endings=[minutes_to_seconds(20),minutes_to_seconds(40)],expected_num_segments=3)
        the_exception = cm.exception
        self.assertIn("segments, got",str(the_exception))

    def test_absorb_intro_repeats(self):
        result = self.do_test(intros=      [3,minutes_to_seconds(1),minutes_to_seconds(30),minutes_to_seconds(30)+2],
                              endings=[minutes_to_seconds(20),minutes_to_seconds(40)],expected_num_segments=2,intro_max_repeat_seconds=60)
        np.testing.assert_array_equal(result,
                                      [
                                        [3,minutes_to_seconds(20)],
                                        [minutes_to_seconds(30),minutes_to_seconds(40)],
                                       ])
        
if __name__ == '__main__':
    unittest.main()