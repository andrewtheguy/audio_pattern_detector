""" # begin new fake news absorption logic      
    def test_absorb_one_short_fake_news(self):
        result = self.process(news_report=[minutes_to_seconds(25),minutes_to_seconds(35),minutes_to_seconds(50)],
                                     intro=[minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)],
                                     )
        np.testing.assert_array_equal(result,[
                                       [minutes_to_seconds(5),minutes_to_seconds(25)],
                                       [minutes_to_seconds(30),minutes_to_seconds(50)],
                                       [minutes_to_seconds(60),self.total_time_1],
                                       ])
        
    def test_absorb_beeps_followed_by_one_short_fake_news(self):
        result = self.process(news_report=[minutes_to_seconds(25),
                                           minutes_to_seconds(25)+1,minutes_to_seconds(25)+2,minutes_to_seconds(25)+3,
                                           minutes_to_seconds(25)+4,minutes_to_seconds(35),
                                           minutes_to_seconds(50)],
                                     intro=[minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)],
                                     )
        np.testing.assert_array_equal(result,[
                                       [minutes_to_seconds(5),minutes_to_seconds(25)],
                                       [minutes_to_seconds(30),minutes_to_seconds(50)],
                                       [minutes_to_seconds(60),self.total_time_1],
                                       ])
        
    def test_not_absorb_more_than_short_fake_news(self):        
        with self.assertRaises(NotImplementedError) as cm:
            result = self.process(news_report=[minutes_to_seconds(25),minutes_to_seconds(35),minutes_to_seconds(36),minutes_to_seconds(50)],
                                        intro=[minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)],
                                        )
            np.testing.assert_array_equal(result,[
                                        [minutes_to_seconds(5),minutes_to_seconds(25)],
                                        [minutes_to_seconds(30),minutes_to_seconds(50)],
                                        [minutes_to_seconds(60),self.total_time_1],
                                        ])
        the_exception = cm.exception
        self.assertIn("not handling news report not followed by intro yet unless news report is 10 seconds",str(the_exception))

        
    def test_not_absorb_short_fake_news_greater_than_next_intro(self):

        with self.assertRaises(NotImplementedError) as cm:
            result = self.process(news_report=[minutes_to_seconds(25),minutes_to_seconds(50),minutes_to_seconds(55)],
                                        intro=[minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)],
                                        )
        the_exception = cm.exception
        self.assertIn("not handling news report not followed by intro yet unless news report is 10 seconds",str(the_exception))

    def test_not_absorb_short_fake_news_not_followed_by_intro(self):

        with self.assertRaises(NotImplementedError) as cm:
            result = self.process(news_report=[minutes_to_seconds(25),minutes_to_seconds(50),minutes_to_seconds(55)],
                                        intro=[minutes_to_seconds(5),minutes_to_seconds(30)],
                                        )
        the_exception = cm.exception
        self.assertIn("not handling news report not followed by intro yet unless news report is 10 seconds",str(the_exception))

        with self.assertRaises(NotImplementedError) as cm:
            result = self.process(news_report=[minutes_to_seconds(25),minutes_to_seconds(50),self.total_time_1],
                                        intro=[minutes_to_seconds(5),minutes_to_seconds(30)],
                                        )
        the_exception = cm.exception
        self.assertIn("not handling news report not followed by intro yet unless news report is 10 seconds",str(the_exception))
        

        with self.assertRaises(NotImplementedError) as cm:
            result = self.process(news_report=[minutes_to_seconds(25),minutes_to_seconds(50),self.total_time_1-20],
                                        intro=[minutes_to_seconds(5),minutes_to_seconds(30)],
                                        )
        the_exception = cm.exception
        self.assertIn("not handling news report not followed by intro yet unless news report is 10 seconds",str(the_exception))

    def test_not_absorb_short_fake_news_at_end(self):
        with self.assertRaises(NotImplementedError) as cm:    
            result = self.process(news_report=[minutes_to_seconds(25),minutes_to_seconds(35),minutes_to_seconds(36),minutes_to_seconds(50),
                                            minutes_to_seconds(80)],
                                        intro=[minutes_to_seconds(5),minutes_to_seconds(30),minutes_to_seconds(60)],
                                        )
        the_exception = cm.exception
        self.assertIn("not handling news report not followed by intro yet unless news report is 10 seconds",str(the_exception))
 """