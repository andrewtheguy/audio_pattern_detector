import unittest
import numpy as np

from process_timestamps import news_intro_cut_off_beginning_and_end
from utils import minutes_to_seconds

class TestConsolidateIntros(unittest.TestCase):
    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)

    def do_test(self,intros,news_reports):
        return news_intro_cut_off_beginning_and_end((list(dict.fromkeys(intros))),(list(dict.fromkeys(news_reports))),self.total_time_1)
    
    def test_zero_everything(self):
        result_news_report = self.do_test(intros=[],
                              news_reports=[])
        np.testing.assert_array_equal(result_news_report,
                                      [])
        
    def test_news_reports_only(self):
        with self.assertRaises(ValueError) as cm:
            result_news_report = self.do_test(intros=[],
                              news_reports=[3,4])
        the_exception = cm.exception
        self.assertIn("cannot have more than one news report within 10 minutes",str(the_exception))

        result_news_report = self.do_test(intros=[],
                              news_reports=[3])
        np.testing.assert_array_equal(result_news_report,
                                      [3])
        
    def test_intros_only(self):
        result_news_report = self.do_test(intros=[3],
                              news_reports=[])
        np.testing.assert_array_equal(result_news_report,
                                      [])
        
        
    def test_too_many_news(self):
        with self.assertRaises(ValueError) as cm:
            result = self.do_test(intros=[3,120],
                                news_reports=[4,5])
        the_exception = cm.exception
        self.assertIn("cannot have more than one news report within 10 minutes",str(the_exception))
        
        with self.assertRaises(ValueError) as cm:
            result = self.do_test(intros=[3,120],
                              news_reports=[4,25])
        the_exception = cm.exception
        self.assertIn("cannot have more than one news report within 10 minutes",str(the_exception))
if __name__ == '__main__':
    unittest.main()