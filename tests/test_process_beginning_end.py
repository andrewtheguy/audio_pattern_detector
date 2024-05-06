import unittest
import numpy as np

from process_timestamps import INTRO_CUT_OFF, news_intro_process_beginning_and_end
from utils import minutes_to_seconds

class TestProcessBeginningAndEndTs(unittest.TestCase):
    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)

    def do_test(self,intros,news_reports):
        return news_intro_process_beginning_and_end((list(dict.fromkeys(intros))),(list(dict.fromkeys(news_reports))),self.total_time_1)
    
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
        
      
    def test_intro_too_late(self):
        with self.assertRaises(ValueError) as cm:
            result = self.do_test(intros=[INTRO_CUT_OFF+10],
                                news_reports=[])
        the_exception = cm.exception
        self.assertIn("first intro cannot be greater than 10 minutes",str(the_exception))
        
        with self.assertRaises(ValueError) as cm:
            result = self.do_test(intros=[INTRO_CUT_OFF+10],
                                news_reports=[4])
        the_exception = cm.exception
        self.assertIn("first intro cannot be greater than 10 minutes",str(the_exception))
        
        with self.assertRaises(ValueError) as cm:
            result = self.do_test(intros=[INTRO_CUT_OFF+10,minutes_to_seconds(30)],
                                news_reports=[INTRO_CUT_OFF+14,minutes_to_seconds(35)])
        the_exception = cm.exception
        self.assertIn("first intro cannot be greater than 10 minutes",str(the_exception))
        
    def test_news_before_intro(self):
        result_news_report = self.do_test(intros=[40],
                              news_reports=[20])
        np.testing.assert_array_equal(result_news_report,
                                      [])
        
        result_news_report = self.do_test(intros=[INTRO_CUT_OFF-10],
                              news_reports=[20])
        np.testing.assert_array_equal(result_news_report,
                                      [])
        
    def test_intro_before_news(self):
        result_news_report = self.do_test(intros=[40,80],
                              news_reports=[60])
        np.testing.assert_array_equal(result_news_report,
                                      [60,self.total_time_1])
        
        
    def test_news_ends_early(self):
        with self.assertRaises(ValueError) as cm:
            result_news_report = self.do_test(intros=[40,INTRO_CUT_OFF+50],
                                news_reports=[INTRO_CUT_OFF+10,INTRO_CUT_OFF+200])
        the_exception = cm.exception
        self.assertIn("cannot end with news reports unless it is within 10 seconds",str(the_exception))
        
    def test_news_not_ends_early(self):
        result_news_report = self.do_test(intros=[40,INTRO_CUT_OFF+50],
                                news_reports=[INTRO_CUT_OFF+10,self.total_time_1])
        np.testing.assert_array_equal(result_news_report,
                                [INTRO_CUT_OFF+10,self.total_time_1])
        
        result_news_report = self.do_test(intros=[40,INTRO_CUT_OFF+50],
                                news_reports=[INTRO_CUT_OFF+10,self.total_time_1-5])
        np.testing.assert_array_equal(result_news_report,
                                [INTRO_CUT_OFF+10,self.total_time_1-5])
        
        
if __name__ == '__main__':
    unittest.main()