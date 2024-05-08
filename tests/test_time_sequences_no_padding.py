import unittest
import numpy as np

from process_timestamps import process_timestamps_rthk
from time_sequence_error import TimeSequenceError
from utils import minutes_to_seconds


class TestProcessTimestamps(unittest.TestCase):
    
    def process(self,news_report,intro,allow_first_short=False):
        return process_timestamps_rthk(news_report,intro,total_time=self.total_time_1,news_report_second_pad=0,allow_first_short=allow_first_short)

    def setUp(self):
        self.total_time_1=minutes_to_seconds(120)

    def test_regular(self):
        result = self.process(news_report=[minutes_to_seconds(21),minutes_to_seconds(73)],
                              intro=[minutes_to_seconds(5),minutes_to_seconds(32),minutes_to_seconds(80)])
        np.testing.assert_array_equal(result,[[minutes_to_seconds(5),minutes_to_seconds(21)],
                                               [minutes_to_seconds(32),minutes_to_seconds(73)],
                                               [minutes_to_seconds(80),self.total_time_1],
                                               ])
        
    def test_middle_the_same(self):
        result = self.process(news_report=[minutes_to_seconds(21),minutes_to_seconds(73)],
                              intro=[minutes_to_seconds(5),minutes_to_seconds(21),minutes_to_seconds(80)])
        np.testing.assert_array_equal(result,[[minutes_to_seconds(5),minutes_to_seconds(21)],
                                               [minutes_to_seconds(21),minutes_to_seconds(73)],
                                               [minutes_to_seconds(80),self.total_time_1],
                                               ])


    def test_zero_everything(self):
        result = self.process(news_report=[],intro=[])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])

    def test_zero_news_report(self):
        result = self.process(news_report=[],intro=[minutes_to_seconds(5),minutes_to_seconds(18)])
        np.testing.assert_array_equal(result,[[minutes_to_seconds(5),self.total_time_1]])
        
        result = self.process(news_report=[],intro=[minutes_to_seconds(5),minutes_to_seconds(20)])
        np.testing.assert_array_equal(result,[[minutes_to_seconds(5),self.total_time_1]])
        
        with self.assertRaises(ValueError) as cm:
            result = self.process(news_report=[],intro=[minutes_to_seconds(11),minutes_to_seconds(30)])
            np.testing.assert_array_equal(result,[[0,self.total_time_1]])

    def test_zero_intro_news_close_to_end(self):
        news_end = self.total_time_1-5
        result = self.process(news_report=[news_end],intro=[])
        np.testing.assert_array_equal(result,[[0,news_end]])
        
    def test_intro_at_10_minutes(self):
        news_end = self.total_time_1-5
        result = self.process(news_report=[news_end],intro=[minutes_to_seconds(10)])
        np.testing.assert_array_equal(result,[[minutes_to_seconds(10),news_end]])
        
    def test_zero_intro_with_news_in_middle(self):
        with self.assertRaises(TimeSequenceError) as cm:
            news_end = self.total_time_1-minutes_to_seconds(20)
            result = self.process(news_report=[news_end],intro=[])
        the_exception = cm.exception
        self.assertIn("cannot end with news reports unless it is within 10 seconds of the end to prevent missing things",str(the_exception))
        
    def test_zero_intro_with_news_at_beginning(self):
        with self.assertRaises(TimeSequenceError) as cm:
            result = self.process(news_report=[0],intro=[])
        the_exception = cm.exception
        self.assertIn("cannot end with news reports unless it is within 10 seconds of the end to prevent missing things",str(the_exception))

    def test_zero_intro_with_multiple_news_in_middle(self):
        with self.assertRaises(TimeSequenceError) as cm:
            news_middle = self.total_time_1-minutes_to_seconds(40)
            news_end = self.total_time_1-minutes_to_seconds(20)
            result = self.process(news_report=[news_middle,news_end],intro=[])
        the_exception = cm.exception
        self.assertIn("cannot end with news reports unless it is within 10 seconds of the end to prevent missing things",str(the_exception))

        with self.assertRaises(ValueError) as cm:
            news_middle = self.total_time_1-minutes_to_seconds(40)
            news_end = self.total_time_1-5
            result = self.process(news_report=[news_middle,news_end],intro=[])
        the_exception = cm.exception
        self.assertIn("intros and news reports must be the same length",str(the_exception))


    # won't cause issues
    def test_news_overflow(self):
        news_end = self.total_time_1+1000
        result = self.process(news_report=[news_end],intro=[])
        np.testing.assert_array_equal(result,[[0,self.total_time_1]])

        result = self.process(news_report=[news_end],intro=[1])
        np.testing.assert_array_equal(result,[[1,self.total_time_1]])

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

    @unittest.skip(reason="it is blocked by other checks now")
    def test_one_intro_same_as_total(self):
        result = self.process(news_report=[]
                                    ,intro=[self.total_time_1])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),self.total_time_1],
                                       ])

    @unittest.skip(reason="it is blocked by other checks now")
    def test_one_intro_same_as_total_with_news(self):
        result = self.process(news_report=[self.total_time_1 - 9]
                                    ,intro=[self.total_time_1])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),self.total_time_1 - 9],
                                       ])
        
    @unittest.skip(reason="it is blocked by other checks now")
    def test_one_intro_same_as_one_news_near_end(self):
        result = self.process(news_report=[self.total_time_1-9]
                                    ,intro=[self.total_time_1-9])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),self.total_time_1-9],
                                       ])
        
    #@unittest.skip(reason="it is blocked by other checks now")    
    def test_one_intro_same_as_one_news_not_near_end(self):
        with self.assertRaises(TimeSequenceError) as cm:
            result = self.process(news_report=[self.total_time_1-11]
                                        ,intro=[self.total_time_1-11])
        #the_exception = cm.exception
        #self.assertIn("first intro cannot be greater than 10 minutes",str(the_exception))

    def test_one_intro_same_as_one_news_at_beginning(self):
        with self.assertRaises(TimeSequenceError) as cm:
            result = self.process(news_report=[9]
                                        ,intro=[9])
        the_exception = cm.exception
        self.assertIn("cannot end with news reports unless it is within 10 seconds of the end to prevent missing things",str(the_exception))

    def test_one_news_report_intro_beginning_news_first(self):
        result = self.process(news_report=[minutes_to_seconds(5)]
                                    ,intro=[minutes_to_seconds(6)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(6),self.total_time_1],
                                       ])


    def test_news_report_ending_as_total_time(self):
        intro1 = minutes_to_seconds(9)
        result = self.process(news_report=[self.total_time_1],intro=[intro1])
        np.testing.assert_array_equal(result,[[intro1,self.total_time_1]])

    def test_news_report_ending_too_early(self):
        intro1 = minutes_to_seconds(9)
        with self.assertRaises(ValueError):
            self.process(news_report=[self.total_time_1-minutes_to_seconds(5)],intro=[intro1])

    def test_news_report_ending_too_early_no_intro(self):
        with self.assertRaises(ValueError):
            self.process(news_report=[self.total_time_1-minutes_to_seconds(5)],intro=[])

    @unittest.skip(reason="reasonable time sequence is always enforced now, which makes this setup always throw timesequenceerror having the beginning too short")
    def test_intro_first_five_minutes_news_less_than_ten(self):
        result = self.process(news_report=[minutes_to_seconds(9),self.total_time_1]
                                    ,intro=[minutes_to_seconds(5),minutes_to_seconds(11)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(5),minutes_to_seconds(9)],
                                       [minutes_to_seconds(11),self.total_time_1]])

    def test_intro_first_five_minutes_news_greater_than_ten(self):
        result = self.process(news_report=[minutes_to_seconds(21),self.total_time_1]
                                    ,intro=[minutes_to_seconds(5),minutes_to_seconds(22)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(5),minutes_to_seconds(21)],
                                       [minutes_to_seconds(22),self.total_time_1]])
        
    @unittest.skip(reason="reasonable time sequence is always enforced now, which makes this setup always throw timesequenceerror having the beginning too short")    
    def test_intro_first_five_minutes_news_earlier(self):
        result = self.process(news_report=[minutes_to_seconds(4),self.total_time_1]
                                    ,intro=[minutes_to_seconds(5),minutes_to_seconds(12)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(4)],
                                       [minutes_to_seconds(5),self.total_time_1]])
        
    @unittest.skip(reason="blocked by intro check")    
    def test_intro_first_after_ten_minutes_intro_first(self):
        result = self.process(news_report=[minutes_to_seconds(20),self.total_time_1]
                                    ,intro=[minutes_to_seconds(11),minutes_to_seconds(30)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(20)],
                                       [minutes_to_seconds(30),self.total_time_1]])
        
    @unittest.skip(reason="reasonable time sequence is always enforced now, which makes this setup always throw timesequenceerror having the beginning too short")    
    def test_intro_first_after_ten_minutes_news_first_before_ten_minutes(self):
        result = self.process(news_report=[minutes_to_seconds(3),minutes_to_seconds(30)]
                                    ,intro=[minutes_to_seconds(13),minutes_to_seconds(40)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(3)],
                                       [minutes_to_seconds(13),minutes_to_seconds(30)],
                                       [minutes_to_seconds(40),self.total_time_1]],
                                       )

    @unittest.skip(reason="it is blocked by other checks now")
    def test_news_report_first_not_absorbed(self):
        result = self.process(news_report=[minutes_to_seconds(23),minutes_to_seconds(53)]
                                    ,intro=[minutes_to_seconds(33),minutes_to_seconds(63)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(0),minutes_to_seconds(23)],
                                       [minutes_to_seconds(33),minutes_to_seconds(53)],
                                       [minutes_to_seconds(63),self.total_time_1],
                                       ])

    def test_news_report_after_news_report(self):
        with self.assertRaises(ValueError):
            result = self.process(news_report=[minutes_to_seconds(22),minutes_to_seconds(32)]
                                        ,intro=[minutes_to_seconds(2),minutes_to_seconds(60)])

    def test_news_report_after_news_report_close_to_end(self):
        with self.assertRaises(ValueError):
            result = self.process(news_report=[minutes_to_seconds(22),minutes_to_seconds(62),self.total_time_1 - 9]
                                        ,intro=[minutes_to_seconds(2),minutes_to_seconds(32)])
            print(result)
    
        
    #  test uneven
    def test_intro_news_report_intro(self):
        result = self.process(news_report=[minutes_to_seconds(19)]
                                    ,intro=[minutes_to_seconds(2),minutes_to_seconds(21)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(2),minutes_to_seconds(19)],
                                       [minutes_to_seconds(21),self.total_time_1],
                                       ])

    def test_multiple_intros(self):
        result = self.process(news_report=[minutes_to_seconds(19)]
                                    ,intro=[minutes_to_seconds(2),minutes_to_seconds(3),minutes_to_seconds(4),minutes_to_seconds(21)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(2),minutes_to_seconds(19)],
                                       [minutes_to_seconds(21),self.total_time_1],
                                       ])

        result = self.process(news_report=[minutes_to_seconds(19),minutes_to_seconds(47)]
                                    ,intro=[minutes_to_seconds(2),minutes_to_seconds(21),
                                            minutes_to_seconds(23),minutes_to_seconds(25),minutes_to_seconds(49)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(2),minutes_to_seconds(19)],
                                       [minutes_to_seconds(21),minutes_to_seconds(47)],
                                       [minutes_to_seconds(49),self.total_time_1],
                                       ])

    def test_middle_same(self):
        with self.assertRaises(ValueError):
            result = self.process(news_report=[minutes_to_seconds(9),minutes_to_seconds(11),]
                                        ,intro=[minutes_to_seconds(2),minutes_to_seconds(9),minutes_to_seconds(13)])

    @unittest.skip(reason="it is always deduplicated now")
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
        result = self.process(news_report=[minutes_to_seconds(51),minutes_to_seconds(21)]
                                    ,intro=[minutes_to_seconds(31),minutes_to_seconds(3),minutes_to_seconds(61)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(3),minutes_to_seconds(21)],
                                       [minutes_to_seconds(31),minutes_to_seconds(51)],
                                       [minutes_to_seconds(61),self.total_time_1],
                                       ])

        

    def test_absorb_first_minute_news_report(self):
        result = self.process(news_report=[30,minutes_to_seconds(28)],
                                     intro=[minutes_to_seconds(8),minutes_to_seconds(30)])
        np.testing.assert_array_equal(result,
                                      [[minutes_to_seconds(8),minutes_to_seconds(28)],
                                       [minutes_to_seconds(30),self.total_time_1],
                                       ])
    
    @unittest.skip(reason="reasonable time sequence is always enforced now, which makes this setup always throw timesequenceerror having the beginning too short")    
    def test_not_absorb_first_minute_news_report_if_intro_comes_first(self):
        result = self.process(news_report=[30,minutes_to_seconds(20)],
                                     intro=[7,minutes_to_seconds(5),minutes_to_seconds(23)],
                                     )
        np.testing.assert_array_equal(result,
                                      [[7,30],
                                       [minutes_to_seconds(5),minutes_to_seconds(20)],
                                       [minutes_to_seconds(23),self.total_time_1],
                                       ])

    def test_allow_first_short_segment(self):
        result = self.process(news_report=[minutes_to_seconds(10),minutes_to_seconds(50)],
                                     intro=[minutes_to_seconds(5),minutes_to_seconds(20),minutes_to_seconds(60)],
                                     allow_first_short=True
                                     )
        np.testing.assert_array_equal(result,[
                                       [minutes_to_seconds(5),minutes_to_seconds(10)],
                                       [minutes_to_seconds(20),minutes_to_seconds(50)],
                                       [minutes_to_seconds(60),self.total_time_1],
                                       ])
        
        result = self.process(news_report=[minutes_to_seconds(15),minutes_to_seconds(50)],
                                     intro=[minutes_to_seconds(5),minutes_to_seconds(20),minutes_to_seconds(60)],
                                     allow_first_short=True
                                     )
        np.testing.assert_array_equal(result,[
                                       [minutes_to_seconds(5),minutes_to_seconds(15)],
                                       [minutes_to_seconds(20),minutes_to_seconds(50)],
                                       [minutes_to_seconds(60),self.total_time_1],
                                       ])
        
        with self.assertRaises(TimeSequenceError):
            result = self.process(news_report=[minutes_to_seconds(15),minutes_to_seconds(30)],
                                        intro=[minutes_to_seconds(5),minutes_to_seconds(20),minutes_to_seconds(40)],
                                        allow_first_short=True
                                        )
            np.testing.assert_array_equal(result,[
                                        [minutes_to_seconds(5),minutes_to_seconds(15)],
                                        [minutes_to_seconds(20),minutes_to_seconds(30)],
                                        [minutes_to_seconds(40),self.total_time_1],
                                        ])
         
         
    
    def test_absorb_beeps(self):
        result = self.process(news_report=[minutes_to_seconds(25),
                                           minutes_to_seconds(25)+1,minutes_to_seconds(25)+2,minutes_to_seconds(25)+3,
                                           minutes_to_seconds(25)+4,
                                           minutes_to_seconds(50)],
                                     intro=[minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)],
                                     )
        np.testing.assert_array_equal(result,[
                                       [minutes_to_seconds(5),minutes_to_seconds(25)],
                                       [minutes_to_seconds(30),minutes_to_seconds(50)],
                                       [minutes_to_seconds(60),self.total_time_1],
                                       ])
    
    def test_absorb_beeps_beginning(self):
        result = self.process(news_report=[minutes_to_seconds(7),
                                           minutes_to_seconds(7)+1,minutes_to_seconds(7)+2,minutes_to_seconds(7)+3,
                                           minutes_to_seconds(7)+4,
                                           minutes_to_seconds(50)],
                                     intro=[minutes_to_seconds(1),minutes_to_seconds(16),minutes_to_seconds(60)],
                                     allow_first_short=True,
                                     )
        np.testing.assert_array_equal(result,[
                                       [minutes_to_seconds(1),minutes_to_seconds(7)],
                                       [minutes_to_seconds(16),minutes_to_seconds(50)],
                                       [minutes_to_seconds(60),self.total_time_1],
                                       ])
    
if __name__ == '__main__':
    unittest.main()