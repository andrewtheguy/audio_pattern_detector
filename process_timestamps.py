from collections import deque
import copy
import logging
import sys

from time_sequence_error import TimeSequenceError
from andrew_utils import seconds_to_time
from utils import is_unique_and_sorted

logger = logging.getLogger(__name__)

# beep 6 times, means 5 repeats
BEEP_PATTERN_REPEAT_LIMIT = 5

def timestamp_sanity_check(result,skip_reasonable_time_sequence_check,allow_first_short=False):
    logger.info(result)
    if(len(result) == 0):
        raise ValueError("result cannot be empty")
    
    for i,r in enumerate(result):
        if(len(r) != 2):
            raise ValueError(f"each element in result must have 2 elements, got {r}")

        beginning = i == 0
        end = i == len(result)-1

        cur_start_time = r[0]
        cur_end_time = r[1]

        if(cur_start_time < 0):
            raise ValueError(f"start time {cur_start_time} is less than 0")
        
        if(cur_start_time > cur_end_time):
            raise ValueError(f"start time {cur_start_time} is greater than end time {cur_end_time}")

        # TODO: still need to account for 1 hour interval news report at night time
        short_allowance_special = 5
        short_allowance_normal = 15
        if not skip_reasonable_time_sequence_check:
            allow_short_interval = allow_first_short and beginning
            # allow first short if intro starts in 2 minutes
            if allow_short_interval and cur_start_time < 2*60 and (cur_end_time - cur_start_time < short_allowance_special*60):
                raise TimeSequenceError(f"duration for program segment {cur_end_time - cur_start_time} seconds is less than {short_allowance_special} minutes for beginning")
            # news report should not last like 15 minutes
            elif not allow_short_interval and cur_end_time - cur_start_time < short_allowance_normal*60:
                raise TimeSequenceError(f"duration for program segment with cur_start_time {seconds_to_time(cur_start_time)} with duration {seconds_to_time(cur_end_time - cur_start_time)} is less than {short_allowance_normal} minutes")
    
    for i in range(1,len(result)):
        cur = result[i]
        cur_start_time = cur[0]
        prev = result[i-1]
        prev_end_time = prev[1]
        gap = cur_start_time - prev_end_time
        if(gap < 0):
            raise ValueError(f"start time {cur_start_time} is less than previous end time {prev_end_time}")
        # news report and commercial time should not be 15 minutes or longer
        elif(not skip_reasonable_time_sequence_check and gap >= 15*60):
            raise TimeSequenceError(f"gap between {cur_start_time} and {prev_end_time} is 15 minutes or longer")
        
    return result

def consolidate_beeps(news_reports):
    if len(news_reports) == 0:
        return news_reports
    if not is_unique_and_sorted(news_reports):
        raise ValueError("news report is not unique or sorted")
    new_ones=[]
    #non_repeating_index = None
    repeat_count = 0
    repeat_limit = BEEP_PATTERN_REPEAT_LIMIT
    for i,cur_news_report in enumerate(news_reports):
        if i == 0:
            #non_repeating_index=i
            new_ones.append(cur_news_report)
        else:
            if repeat_count < repeat_limit and cur_news_report - news_reports[i-1] <= 2: #seconds
                repeat_count += 1
            else:
                repeat_count = 0
                #non_repeating_index=i
                new_ones.append(cur_news_report)
    return new_ones            
        
# news_reports need to be unique         
def consolidate_intros(intros,news_reports):
    if not is_unique_and_sorted(intros):
        raise ValueError("intros is not unique or sorted")
    if not is_unique_and_sorted(news_reports):
        raise ValueError("news report is not unique or sorted")
    
    consolidated_intros = []
    intro_index = 0
    news_index = 0
    
    #edge case
    if len(news_reports) == 0:
        #unique
        return list(dict.fromkeys(intros))
    
    news_reports=news_reports.copy()
    
    #normalize
    if(len(intros) > 0):
        if(intros[-1]>news_reports[-1]):
            news_reports.append(sys.maxsize)
    
    
    while intro_index < len(intros) and news_index < len(news_reports):
        intro = intros[intro_index]
        news = news_reports[news_index]

        if intro <= news:
            consolidated_intros.append(intro)

        # Check if there are extra intros before the current news
        while intro_index < len(intros) - 1 and intros[intro_index + 1] < news:
            intro_index += 1  # Skip extra intros

        intro_index += 1
        news_index += 1  # Move to the next news report

    print("intro_index",intro_index)
    print("news_index",news_index)
    # leftover intros
    if(intro_index < len(intros) and intros[intro_index]>=news_reports[-1]):
        consolidated_intros.append(intros[intro_index])
        
    # leftover news reports    
    #if(news_index < len(news_reports) and news_reports[news_index]>=intros[-1]):
    #    consolidated_intros.append(intros[-1])
        
    return consolidated_intros
    

def process_timestamps(news_reports,intros,total_time,news_report_second_pad=6,
                       allow_first_short=False):

    skip_reasonable_time_sequence_check=False


    if len(news_reports) != len(set(news_reports)):
       raise ValueError("news report has duplicates, clean up duplicates first")   

    if len(intros) != len(set(intros)):
       raise ValueError("intro has duplicates, clean up duplicates first")   

    for ts in intros:
        if ts > total_time:
            raise ValueError(f"intro overflow, is greater than total time {total_time}")
        elif ts < 0:
            raise ValueError(f"intro is less than 0")


    # will bug out if not sorted
    # TODO maybe require input to be sorted first to prevent
    # sorting inputs that are already sorted again
    #news_report = deque([40,90,300])
    #intro =       deque([60,200,400])
    news_reports=deque(sorted(news_reports))
    intros=deque(sorted(intros))


    cur_intro = 0

    #news_report = deque([598, 2398, 3958, 5758])
    #intro = deque([1056, 2661, 4463])
    # no news report
    if(len(news_reports)==0):
        # no need to trim
        return [[cur_intro, total_time]]
    
    # news report within the first 1 minute and it is less than the first intro, change to 0
    if(len(intros) > 0 and (news_reports[0] <= 1*60 and news_reports[0] < intros[0])):
        news_reports[0]=0

    news_reports=deque(news_reports)
    intros=deque(intros)

    # intro starts before news report,
    # shift cur_intro from 0 to the first intro
    # if it is less than 10 minutes,
    # it is very unlikely to miss a news report
    # within the first 10 minutes and at the same time
    # the program has already started before 10 minutes
    if(len(intros) > 0 and intros[0] <= 10*60 and intros[0] < news_reports[0]):
        cur_intro = intros.popleft()

    if(cur_intro > total_time):
        raise ValueError("intro overflow, is greater than total time {total_time}")

    

    pair=[]
    #print("fgfdgdfgfdgdfgfd")
    news_report_followed_by_intro = True
    while(len(news_reports)>0):
        if(not news_report_followed_by_intro):
           raise ValueError("cannot have news report followed by news report")
        news_report_followed_by_intro=False
        cur_news_report = news_reports.popleft()
        if(cur_intro > total_time):
            raise ValueError(f"intro overflow, is greater than total time {total_time}")
        # clean up beep beep beep
        max_beep_repeat = 10
        count_beep_repeat = 0
        #print("cur_news_report",cur_news_report)
        #print("news_report[0] - cur_news_report",news_report[0] - cur_news_report)
        beep_tracker=cur_news_report
        while len(news_reports)>0 and news_reports[0] - beep_tracker <= 10 and count_beep_repeat < max_beep_repeat:
            beep_tracker=news_reports.popleft()
            count_beep_repeat += 1
        # absorb fake news report beep within 16 minutes of intro except allow short intro or news report not followed by intro
        # or absorbtion would cause too long
        if len(pair) == 0:
            pass # no absorption for first pair because it is error prone
        else:
            #target_min_intro_duration = 18
            # pop only one within 15 minutes and 30 seconds   
            if len(intros) > 0 and len(news_reports)>0 and cur_news_report <= cur_intro + 15*60+30 and cur_news_report < intros[0]:
                cur_news_report=news_reports.popleft()
        pair.append([cur_intro, cur_news_report])
        # get first intro after news report while ignoring others after first
        while(len(intros)>0):
             cur_intro = intros.popleft()
             if cur_intro > cur_news_report:
                 # ends with intro but no news report
                 if len(news_reports)==0:
                    pair.append([cur_intro, total_time])

                 if(len(news_reports)>0 and cur_intro > news_reports[0]):
                    # intro greater than two news reports, which means it is news report followed by news report
                    # will cause start time to be greater than end time for the next time range to be added
                    news_report_followed_by_intro=False
                 else:    
                    news_report_followed_by_intro=True    
                 break
        # prevent missing something in the middle     
        # unlkely to happen if news report is 10 seconds from the end w/o intro
        if not news_report_followed_by_intro and cur_news_report <= total_time - 10:
            raise NotImplementedError(f"not handling news report not followed by intro yet unless news report is 10 seconds from the end to prevent missing an intro, cur_news_report {cur_news_report}, cur_intro: {cur_intro}")
    #print("before padding",pair)
    for i,arr in enumerate(pair):
        cur_intro = arr[0]
        cur_news_report = arr[1]
        if(i+1>=len(pair)):
            next_intro = None
        else:
            next_intro = pair[i+1][0]
        # pad news_report_second_pad seconds to news report if it is larger then news_report_second_pad
        if((next_intro is None or (cur_news_report + news_report_second_pad <= next_intro and cur_news_report>news_report_second_pad)) and cur_news_report < total_time):
            arr[1] = cur_news_report + news_report_second_pad
    #print("after padding",pair)

    # remove start = end
    result = list(filter(lambda x: (x[0] != x[1]), pair)) 

    #required sanity check
    if(len(result) == 0):
        raise ValueError("result cannot be empty")
    
    timestamp_sanity_check(result,skip_reasonable_time_sequence_check=skip_reasonable_time_sequence_check,allow_first_short=allow_first_short)

    return result
