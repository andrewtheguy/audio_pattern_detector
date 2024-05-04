from scrape import process_timestamps, timestamp_sanity_check

import unittest
import numpy as np

from time_sequence_error import TimeSequenceError

def minutes_to_seconds(minutes):
    return minutes*60


def hours_to_seconds(hours):
    return minutes_to_seconds(hours*60)

class TestProcessTimestamps(unittest.TestCase):
    
    def process(self,news_report,intro):
        return process_timestamps(news_report,intro,total_time=self.total_time_1,news_report_second_pad=0,skip_reasonable_time_sequence_check=True)

    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)

    def test_zero_everything(self):
        result = self.process(news_report=[],intro=[])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])

    def test_zero_news_report(self):
        result = self.process(news_report=[],intro=[minutes_to_seconds(5),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])
        result = self.process(news_report=[],intro=[minutes_to_seconds(5),minutes_to_seconds(9)])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])
        result = self.process(news_report=[],intro=[minutes_to_seconds(11),minutes_to_seconds(12)])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])

    def test_zero_intro(self):
        news_end = self.total_time_1-5
        result = self.process(news_report=[news_end],intro=[])
        np.testing.assert_array_equal(result,[[0,news_end]])

    # won't cause issues
    def test_news_overflow(self):
        news_end = self.total_time_1+1000
        result = self.process(news_report=[news_end],intro=[])
        np.testing.assert_array_equal(result,[[0,news_end]])

        result = self.process(news_report=[news_end],intro=[1])
        np.testing.assert_array_equal(result,[[1,news_end]])

    def test_intro_overflow(self):
        with self.assertRaises(ValueError) as cm:
            intro = self.total_time_1+500
            news_end = self.total_time_1+1000
            result = self.process(news_report=[news_end],intro=[intro])
        the_exception = cm.exception
        self.assertIn("intro overflow, is greater than total time",str(the_exception))

        with self.assertRaises(ValueError):
            result = self.process(news_report=[self.total_time_1+10],intro=[self.total_time_1+1500])
        the_exception = cm.exception
        self.assertIn("intro overflow, is greater than total time",str(the_exception))

        with self.assertRaises(ValueError):
            result = self.process(news_report=[10,self.total_time_1+10],intro=[5,self.total_time_1+1500])
        the_exception = cm.exception
        self.assertIn("intro overflow, is greater than total time",str(the_exception))

    def test_one_intro_same_as_total(self):
        result = self.process(news_report=[]
                                    ,intro=[self.total_time_1])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),self.total_time_1],
                                       ])

    def test_one_intro_same_as_total_with_news(self):
        result = self.process(news_report=[self.total_time_1 - 9]
                                    ,intro=[self.total_time_1])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),self.total_time_1 - 9],
                                       ])
        
            
    def test_one_intro_same_as_one_news_near_end(self):
        result = self.process(news_report=[self.total_time_1-9]
                                    ,intro=[self.total_time_1-9])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),self.total_time_1-9],
                                       ])
        
    def test_one_intro_same_as_one_news_not_near_end(self):
        with self.assertRaises(NotImplementedError) as cm:
            result = self.process(news_report=[self.total_time_1-11]
                                        ,intro=[self.total_time_1-11])
        the_exception = cm.exception
        self.assertIn("not handling news report not followed by intro yet unless news report is 10 seconds",str(the_exception))

    def test_one_intro_same_as_one_news_at_beginning(self):
        with self.assertRaises(NotImplementedError) as cm:
            result = self.process(news_report=[9]
                                        ,intro=[9])
        the_exception = cm.exception
        self.assertIn("not handling news report not followed by intro yet unless news report is 10 seconds",str(the_exception))

    def test_one_news_report_intro(self):
        result = self.process(news_report=[minutes_to_seconds(11)]
                                    ,intro=[minutes_to_seconds(12)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(11)],
                                       [minutes_to_seconds(12),self.total_time_1],
                                       ])


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

    def test_intro_first_five_minutes_news_greater_than_ten(self):
        result = self.process(news_report=[minutes_to_seconds(11),self.total_time_1]
                                    ,intro=[minutes_to_seconds(5),minutes_to_seconds(12)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(5),minutes_to_seconds(11)],
                                       [minutes_to_seconds(12),self.total_time_1]])
        
    def test_intro_first_five_minutes_news_earlier(self):
        result = self.process(news_report=[minutes_to_seconds(4),self.total_time_1]
                                    ,intro=[minutes_to_seconds(5),minutes_to_seconds(12)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(4)],
                                       [minutes_to_seconds(5),self.total_time_1]])

    def test_intro_first_after_ten_minutes_intro_first(self):
        result = self.process(news_report=[minutes_to_seconds(20),self.total_time_1]
                                    ,intro=[minutes_to_seconds(11),minutes_to_seconds(30)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(20)],
                                       [minutes_to_seconds(30),self.total_time_1]])
        
    def test_intro_first_after_ten_minutes_news_first(self):
        result = self.process(news_report=[minutes_to_seconds(3),minutes_to_seconds(30)]
                                    ,intro=[minutes_to_seconds(13),minutes_to_seconds(40)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(3)],
                                       [minutes_to_seconds(13),minutes_to_seconds(30)],
                                       [minutes_to_seconds(40),self.total_time_1]],
                                       )

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
        
    #  test uneven
    def test_intro_news_report_intro(self):
        result = self.process(news_report=[minutes_to_seconds(9)]
                                    ,intro=[minutes_to_seconds(2),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(2),minutes_to_seconds(9)],
                                       [minutes_to_seconds(11),self.total_time_1],
                                       ])

    def test_multiple_intros(self):
        result = self.process(news_report=[minutes_to_seconds(9)]
                                    ,intro=[minutes_to_seconds(2),minutes_to_seconds(3),minutes_to_seconds(4),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(2),minutes_to_seconds(9)],
                                       [minutes_to_seconds(11),self.total_time_1],
                                       ])

        result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(17)]
                                    ,intro=[minutes_to_seconds(2),minutes_to_seconds(11),
                                            minutes_to_seconds(13),minutes_to_seconds(15),minutes_to_seconds(19)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(2),minutes_to_seconds(9)],
                                       [minutes_to_seconds(11),minutes_to_seconds(17)],
                                       [minutes_to_seconds(19),self.total_time_1],
                                       ])

    def test_middle_same(self):
        with self.assertRaises(NotImplementedError):
            result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(11),]
                                        ,intro=[minutes_to_seconds(2),minutes_to_seconds(9),minutes_to_seconds(13)])

    def test_duplicates(self):
        with self.assertRaises(ValueError) as cm:
            result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(9),minutes_to_seconds(9),minutes_to_seconds(17),minutes_to_seconds(17)]
                                        ,intro=[minutes_to_seconds(2),minutes_to_seconds(11),minutes_to_seconds(11),minutes_to_seconds(11),minutes_to_seconds(11),
                                                minutes_to_seconds(13),minutes_to_seconds(15),minutes_to_seconds(19)])
        the_exception = cm.exception
        self.assertIn("news report has duplicates",str(the_exception))
        with self.assertRaises(ValueError) as cm:
            result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(17)]
                                        ,intro=[minutes_to_seconds(2),minutes_to_seconds(11),minutes_to_seconds(11),minutes_to_seconds(11),minutes_to_seconds(11),
                                                minutes_to_seconds(13),minutes_to_seconds(15),minutes_to_seconds(19)])
        the_exception = cm.exception
        self.assertIn("intro has duplicates",str(the_exception))

    def test_out_of_order(self):
        result = self.process(news_report=[minutes_to_seconds(20),minutes_to_seconds(9)]
                                    ,intro=[minutes_to_seconds(10),minutes_to_seconds(3),minutes_to_seconds(27)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(3),minutes_to_seconds(9)],
                                       [minutes_to_seconds(10),minutes_to_seconds(20)],
                                       [minutes_to_seconds(27),self.total_time_1],
                                       ])

        result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(17)]
                                    ,intro=[minutes_to_seconds(2),minutes_to_seconds(11),
                                            minutes_to_seconds(13),minutes_to_seconds(15),minutes_to_seconds(19)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(2),minutes_to_seconds(9)],
                                       [minutes_to_seconds(11),minutes_to_seconds(17)],
                                       [minutes_to_seconds(19),self.total_time_1],
                                       ])
        

    def test_absorb_first_minute_news_report(self):
        result = self.process(news_report=[30,minutes_to_seconds(15)],
                                     intro=[minutes_to_seconds(8),minutes_to_seconds(18)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(8),minutes_to_seconds(15)],
                                       [minutes_to_seconds(18),self.total_time_1],
                                       ])
        
    def test_not_absorb_first_minute_news_report(self):
        result = self.process(news_report=[30,minutes_to_seconds(20)],
                                     intro=[7,minutes_to_seconds(5),minutes_to_seconds(23)],
                                     )
        np.testing.assert_array_equal(result,
                                      [[7,30],
                                       [minutes_to_seconds(5),minutes_to_seconds(20)],
                                       [minutes_to_seconds(23),self.total_time_1],
                                       ])
          
        
        


class TestProcessTimestampsWithPadding(unittest.TestCase):
    
    def process(self,news_report,intro):
        return process_timestamps(news_report,intro,total_time=self.total_time_1,news_report_second_pad=self.news_report_second_pad,skip_reasonable_time_sequence_check=True)

    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)
        self.news_report_second_pad=10

    def test_zero_everything(self):
        result = self.process(news_report=[],intro=[])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])

    def test_zero_news_report(self):
        result = self.process(news_report=[],intro=[minutes_to_seconds(5),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])
        result = self.process(news_report=[],intro=[minutes_to_seconds(5),minutes_to_seconds(9)])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])
        result = self.process(news_report=[],intro=[minutes_to_seconds(11),minutes_to_seconds(12)])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])

    def test_no_overlap(self):
        result = self.process(news_report=[minutes_to_seconds(11)]
                                    ,intro=[minutes_to_seconds(8),minutes_to_seconds(13)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(8),minutes_to_seconds(11)+self.news_report_second_pad],
                                        [minutes_to_seconds(13),self.total_time_1],
                                      ])
        
    def test_middle_same(self):
        with self.assertRaises(NotImplementedError):
            result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(11),]
                                        ,intro=[minutes_to_seconds(2),minutes_to_seconds(9),minutes_to_seconds(13)])

    def test_overlap(self):
        result = self.process(news_report=[minutes_to_seconds(11),minutes_to_seconds(13)]
                                    ,intro=[minutes_to_seconds(8),minutes_to_seconds(11)+2,minutes_to_seconds(15)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(8),minutes_to_seconds(11)],
                                        [minutes_to_seconds(11)+2,minutes_to_seconds(13)+self.news_report_second_pad],
                                        [minutes_to_seconds(15),self.total_time_1],
                                      ])

    def test_news_report_beginning(self):
        result = self.process(news_report=[minutes_to_seconds(0),minutes_to_seconds(13)]
                                    ,intro=[minutes_to_seconds(8),minutes_to_seconds(15)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(8),minutes_to_seconds(13)+self.news_report_second_pad],
                                        [minutes_to_seconds(15),self.total_time_1],
                                      ])
        
    def test_news_report_beginning_same_as_padding(self):
        result = self.process(news_report=[self.news_report_second_pad,minutes_to_seconds(30)]
                                    ,intro=[minutes_to_seconds(20),minutes_to_seconds(40)])
        np.testing.assert_array_equal(result,
                                      [
                                          [minutes_to_seconds(20),minutes_to_seconds(30)+self.news_report_second_pad],
                                        [minutes_to_seconds(40),self.total_time_1],
                                      ])
        
    def test_intro_earlier(self):
        result = self.process(news_report=[self.news_report_second_pad,minutes_to_seconds(30)]
                                    ,intro=[self.news_report_second_pad-2,minutes_to_seconds(20),minutes_to_seconds(40)])
        np.testing.assert_array_equal(result,
                                      [
                                          [self.news_report_second_pad-2,self.news_report_second_pad],
                                          [minutes_to_seconds(20),minutes_to_seconds(30)+self.news_report_second_pad],
                                        [minutes_to_seconds(40),self.total_time_1],
                                      ])


class TestDurationAndGaps(unittest.TestCase):
    
    def check(self,result,allow_first_short=False):
        return timestamp_sanity_check(result,skip_reasonable_time_sequence_check=False,allow_first_short=allow_first_short)

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            result = self.check([])

    def test_empty_2d(self):
        with self.assertRaises(ValueError):
            result = self.check([[]])

    def test_valid(self):
        try:
            result = self.check([[1,minutes_to_seconds(18)]])
        except Exception as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")
        
    def test_1_value(self):
        with self.assertRaises(ValueError):
            result = self.check([[1]])

    def test_3_values(self):
        with self.assertRaises(ValueError):
            result = self.check([[1,2,3]])

    def test_valid(self):
        try:
            result = self.check([[1,minutes_to_seconds(18)]])
        except Exception as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")

    def test_too_short(self):
        with self.assertRaises(TimeSequenceError):
            result = self.check([[1,minutes_to_seconds(14)]])

    def test_allow_first_short(self):
        #with self.assertRaises(TimeSequenceError):
        result = self.check([[1,minutes_to_seconds(6)]],allow_first_short=True)
        with self.assertRaises(TimeSequenceError):
            result = self.check([
                [1,minutes_to_seconds(13)],
                [minutes_to_seconds(14), minutes_to_seconds(18)],
            ],
            allow_first_short=True)

    def test_allow_first_short_only_close_enough_to_beginning(self):
        with self.assertRaises(TimeSequenceError):
            result = self.check([
                [4,minutes_to_seconds(9)],
                [minutes_to_seconds(12), minutes_to_seconds(30)],
            ])

    def test_gap_normal(self):
        try:
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(25),minutes_to_seconds(50)]])
        except TimeSequenceError as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")

    def test_allow_continuous(self):
        try:
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(20),minutes_to_seconds(50)]])
        except Exception as e:  
            self.fail(f"myFunc() raised {type(e)}: {e} unexpectedly!")

    def test_gap_too_large(self):
        with self.assertRaises(TimeSequenceError):
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(50),minutes_to_seconds(60)]])

    def test_disallow_flip_over(self):
        with self.assertRaises(ValueError):
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(19),minutes_to_seconds(60)]])

    def test_disallow_negative(self):
        with self.assertRaises(ValueError):
            result = self.check([[-1,minutes_to_seconds(20)],[minutes_to_seconds(25),minutes_to_seconds(50)]])

    def test_disallow_overlap(self):
        with self.assertRaises(ValueError):
            result = self.check([[1,minutes_to_seconds(20)],[minutes_to_seconds(19),minutes_to_seconds(50)]])

if __name__ == '__main__':
    unittest.main()