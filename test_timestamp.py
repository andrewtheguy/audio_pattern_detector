from scrape import process_timestamps

import unittest
import numpy as np

def minutes_to_seconds(minutes):
    return minutes*60


def hours_to_seconds(hours):
    return minutes_to_seconds(hours*60)

class TestProcessTimestamps(unittest.TestCase):
    
    def process(self,news_report,intro):
        return process_timestamps(news_report,intro,total_time=self.total_time_1,news_report_second_pad=0)

    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)

    def test_zero_everything(self):
        result = self.process(news_report=[],intro=[])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])

    def test_zero_news_report(self):
        result = self.process(news_report=[],intro=[minutes_to_seconds(5),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])

    def test_zero_intro(self):
        news_end = self.total_time_1-5
        result = self.process(news_report=[news_end],intro=[])
        np.testing.assert_array_equal(result,[[0,news_end]])


    def test_news_report_ending_as_total_time(self):
        intro1 = minutes_to_seconds(9)
        result = self.process(news_report=[self.total_time_1],intro=[intro1])
        np.testing.assert_array_equal(result,[[intro1,self.total_time_1]])

    def test_news_report_ending_too_early(self):
        intro1 = minutes_to_seconds(9)
        with self.assertRaises(NotImplementedError):
            self.process(news_report=[self.total_time_1-minutes_to_seconds(5)],intro=[intro1])

    def test_news_report_ending_too_early_no_intro(self):
        with self.assertRaises(NotImplementedError):
            self.process(news_report=[self.total_time_1-minutes_to_seconds(5)],intro=[])

    def test_intro_first_five_minutes_news_less_than_ten(self):
        result = self.process(news_report=[minutes_to_seconds(9),self.total_time_1]
                                    ,intro=[minutes_to_seconds(5),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(5),minutes_to_seconds(9)],
                                       [minutes_to_seconds(11),self.total_time_1]])

    def test_intro_first_after_ten_minutes(self):
        result = self.process(news_report=[minutes_to_seconds(20),self.total_time_1]
                                    ,intro=[minutes_to_seconds(11),minutes_to_seconds(30)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(20)],
                                       [minutes_to_seconds(30),self.total_time_1]])

    def test_news_report_first(self):
        result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(20)]
                                    ,intro=[minutes_to_seconds(11),minutes_to_seconds(35)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(9)],
                                       [minutes_to_seconds(11),minutes_to_seconds(20)],
                                       [minutes_to_seconds(35),self.total_time_1],
                                       ])

    def test_news_report_after_news_report(self):
        with self.assertRaises(NotImplementedError):
            result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(20)]
                                        ,intro=[minutes_to_seconds(2),minutes_to_seconds(50)])

    def test_news_report_after_news_report_close_to_end(self):
        result = self.process(news_report=[minutes_to_seconds(9),self.total_time_1 - 9]
                                    ,intro=[minutes_to_seconds(2),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(2),minutes_to_seconds(9)],
                                       [minutes_to_seconds(11),self.total_time_1 - 9],
                                       ])
        
    # begin test uneven
    def test_intro_news_report_intro(self):
        result = self.process(news_report=[minutes_to_seconds(9)]
                                    ,intro=[minutes_to_seconds(2),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(2),minutes_to_seconds(9)],
                                       [minutes_to_seconds(11),self.total_time_1],
                                       ])


if __name__ == '__main__':
    unittest.main()