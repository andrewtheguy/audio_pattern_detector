import unittest
import numpy as np

from process_timestamps import INTRO_CUT_OFF, absorb_fake_news_report, build_time_sequence
from time_sequence_error import TimeSequenceError
from utils import minutes_to_seconds

class TestAbsorbFakeNews(unittest.TestCase):

    def do_test(self,intros,news_reports):
        return absorb_fake_news_report(intros,news_reports)
    
    def test_no_absorb_more_intros_than_news(self):
        intros=      [minutes_to_seconds(5),minutes_to_seconds(6),minutes_to_seconds(30),minutes_to_seconds(90)]
        news_reports=[minutes_to_seconds(25),minutes_to_seconds(35),minutes_to_seconds(50)]
        result = self.do_test(news_reports=news_reports,
                                     intros=intros,
                                     )
        np.testing.assert_array_equal(result,
                                       news_reports,
                                       ) 

    def test_no_absorb_same_length(self):
        intros=      [minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(90)]
        news_reports=[minutes_to_seconds(25),minutes_to_seconds(35),minutes_to_seconds(50)]
        result = self.do_test(news_reports=news_reports,
                                     intros=intros,
                                     )
        np.testing.assert_array_equal(result,
                                       news_reports,
                                       )
        
    def test_no_absorb_in_between(self):
        intros=      [minutes_to_seconds(5),minutes_to_seconds(30),                        minutes_to_seconds(65)]
        news_reports=[minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(60),minutes_to_seconds(90)]
        result = self.do_test(news_reports=news_reports,
                                     intros=intros,
                                     )
        np.testing.assert_array_equal(result,
                                       news_reports,
                                       )
        
    def test_absorb_one(self):
        intros=      [minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)]
        news_reports=[minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(65),minutes_to_seconds(85)]
        result = self.do_test(news_reports=news_reports,
                                     intros=intros,
                                     )
        np.testing.assert_array_equal(result,
                                       [minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(85)],
                                       )
        
    def test_not_absorb_if_results_to_long_duration(self):
        intros=      [minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)]
        news_reports=[minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(65),minutes_to_seconds(100)]
        result = self.do_test(news_reports=news_reports,
                                     intros=intros,
                                     )
        np.testing.assert_array_equal(result,news_reports)
        
    def test_absorb_only_one(self):
        intros=      [minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)]
        news_reports=[minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(65),minutes_to_seconds(66),minutes_to_seconds(85)]
        result = self.do_test(news_reports=news_reports,
                                     intros=intros,
                                     )
        np.testing.assert_array_equal(result,
                                       [minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(66),minutes_to_seconds(85)],
                                       )
        
    def test_absorb_only_one_second_close_by(self):
        intros=      [minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)]
        news_reports=[minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(65),minutes_to_seconds(65)+2,minutes_to_seconds(85)]
        result = self.do_test(news_reports=news_reports,
                                     intros=intros,
                                     )
        np.testing.assert_array_equal(result,
                                       [minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(65)+2,minutes_to_seconds(85)],
                                       )

    # def test_absorb_two_when_second_one_close_by(self):
    #     intros=      [minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)]
    #     news_reports=[minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(65),minutes_to_seconds(65)+2,minutes_to_seconds(85)]
    #     result = self.do_test(news_reports=news_reports,
    #                                  intros=intros,
    #                                  )
    #     np.testing.assert_array_equal(result,
    #                                    [minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(85)],
    #                                    )

    # def test_absorb_only_two_when_second_one_close_by(self):
    #     intros=      [minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)]
    #     news_reports=[minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(65),minutes_to_seconds(65)+2,minutes_to_seconds(66),minutes_to_seconds(85)]
    #     result = self.do_test(news_reports=news_reports,
    #                                  intros=intros,
    #                                  )
    #     np.testing.assert_array_equal(result,
    #                                    [minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(66),minutes_to_seconds(85)],
    #                                    )

    def test_not_absorb_negative(self):
        intros=      [minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)]
        news_reports=[minutes_to_seconds(25),minutes_to_seconds(55),minutes_to_seconds(56),minutes_to_seconds(85)]
        result = self.do_test(news_reports=news_reports,
                                     intros=intros,
                                     )
        np.testing.assert_array_equal(result,
                                       news_reports,
                                       )