import unittest
import numpy as np

from process_timestamps import process_timestamps
from utils import minutes_to_seconds
from time_sequence_error import TimeSequenceError


class TestProcessTimestampsWithPadding(unittest.TestCase):

    def process(self,news_report,intro):
        return process_timestamps(news_report,intro,total_time=self.total_time_1,news_report_second_pad=self.news_report_second_pad)

    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)
        self.news_report_second_pad=10

    def test_zero_everything(self):
        result = self.process(news_report=[],intro=[])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])

    def test_zero_news_report(self):
        result = self.process(news_report=[],intro=[minutes_to_seconds(5),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,[[minutes_to_seconds(5),self.total_time_1]])
        result = self.process(news_report=[],intro=[minutes_to_seconds(5),minutes_to_seconds(9)])
        np.testing.assert_array_equal(result,[[minutes_to_seconds(5),self.total_time_1]])
        with self.assertRaises(ValueError) as cm:
          result = self.process(news_report=[],intro=[minutes_to_seconds(11),minutes_to_seconds(12)])
          np.testing.assert_array_equal(result,[[0,self.total_time_1]])
        the_exception = cm.exception
        self.assertIn("first intro cannot be greater than 10 minutes",str(the_exception))


    def test_no_overlap(self):
        result = self.process(news_report=[minutes_to_seconds(30)]
                                    ,intro=[minutes_to_seconds(8),minutes_to_seconds(35)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(8),minutes_to_seconds(30)+self.news_report_second_pad],
                                        [minutes_to_seconds(35),self.total_time_1],
                                      ])
        
    def test_middle_same(self):
        with self.assertRaises(TimeSequenceError):
            result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(11),]
                                        ,intro=[minutes_to_seconds(2),minutes_to_seconds(9),minutes_to_seconds(13)])

    def test_overlap(self):
        result = self.process(news_report=[minutes_to_seconds(24),minutes_to_seconds(48)]
                                    ,intro=[minutes_to_seconds(8),minutes_to_seconds(24)+2,minutes_to_seconds(50)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(8),minutes_to_seconds(24)+2],
                                        [minutes_to_seconds(24)+2,minutes_to_seconds(48)+self.news_report_second_pad],
                                        [minutes_to_seconds(50),self.total_time_1],
                                      ])

    def test_news_report_beginning(self):
        result = self.process(news_report=[minutes_to_seconds(0),minutes_to_seconds(28)]
                                    ,intro=[minutes_to_seconds(8),minutes_to_seconds(30)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(8),minutes_to_seconds(28)+self.news_report_second_pad],
                                        [minutes_to_seconds(30),self.total_time_1],
                                      ])
        
    def test_news_report_beginning_same_as_padding(self):
        result = self.process(news_report=[self.news_report_second_pad,minutes_to_seconds(50)]
                                    ,intro=[minutes_to_seconds(10),minutes_to_seconds(60)])
        np.testing.assert_array_equal(result,
                                      [
                                        [minutes_to_seconds(10),minutes_to_seconds(50)+self.news_report_second_pad],
                                        [minutes_to_seconds(60),self.total_time_1],
                                      ])
        
    @unittest.skip(reason="reasonable time sequence is always enforced now, which makes this setup always throw timesequenceerror having the beginning too short")
    def test_intro_earlier_but_still_within_padding(self):
        result = self.process(news_report=[self.news_report_second_pad,minutes_to_seconds(30)]
                                    ,intro=[self.news_report_second_pad-2,minutes_to_seconds(20),minutes_to_seconds(40)])
        np.testing.assert_array_equal(result,
                                      [
                                          [self.news_report_second_pad-2,self.news_report_second_pad],
                                          [minutes_to_seconds(20),minutes_to_seconds(30)+self.news_report_second_pad],
                                        [minutes_to_seconds(40),self.total_time_1],
                                      ])
