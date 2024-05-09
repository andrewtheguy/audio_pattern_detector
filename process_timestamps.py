from collections import deque
import copy
import logging
import math
import sys

from time_sequence_error import TimeSequenceError
from andrew_utils import seconds_to_time
from utils import is_unique_and_sorted

logger = logging.getLogger(__name__)

# allow 7 seconds of beeps to repeat
BEEP_PATTERN_REPEAT_SECONDS = 7
# allow one intro and news report within 10 minutes
# but not intro past 10 minutes
INTRO_CUT_OFF=10*60

def timestamp_sanity_check(result,total_time):

    if(len(result) == 0):
        raise ValueError("result cannot be empty")
    
    for i,r in enumerate(result):
        if(len(r) != 2):
            raise ValueError(f"each element in result must have 2 elements, got {r}")

        beginning = i == 0
        # allow end to be larger than total time for now
        end = i == len(result)-1

        cur_start_time = r[0]
        cur_end_time = r[1]

        if cur_start_time >= total_time:
            raise ValueError(f"start time overflow, is greater or equals to total time {total_time}")
        elif(cur_end_time == cur_start_time):
            raise ValueError(f"start time is equal to end time, did you forget to remove zero duration sequences first?")
        elif(cur_start_time < 0):
            raise ValueError(f"start time {cur_start_time} is less than 0")
        
        if(cur_start_time > cur_end_time):
            raise ValueError(f"start time {cur_start_time} is greater than end time {cur_end_time}")

    for i in range(1,len(result)):
        cur = result[i]
        cur_start_time = cur[0]
        prev = result[i-1]
        prev_end_time = prev[1]
        gap = cur_start_time - prev_end_time
        if(gap < 0):
            raise ValueError(f"start time {cur_start_time} is less than previous end time {prev_end_time}")
   
    return result

def timestamp_sanity_check_rthk(result,total_time,allow_first_short=False):
    timestamp_sanity_check(result,total_time)
    
    for i,r in enumerate(result):
        if(len(r) != 2):
            raise ValueError(f"each element in result must have 2 elements, got {r}")

        beginning = i == 0
        # allow end to be larger than total time for now
        end = i == len(result)-1

        cur_start_time = r[0]
        cur_end_time = r[1]
        
        # TODO: still need to account for 1 hour interval news report at night time
        short_allowance_special = 5
        short_allowance_normal = 15
        
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
        elif(gap >= 15*60):
            raise TimeSequenceError(f"gap between {cur_start_time} and {prev_end_time} is 15 minutes or longer")
        
    return result

def preprocess_ts(peak_times,remove_repeats=False,max_repeat_seconds=None):
    # deduplicate by seconds
    # sort: will bug out if not sorted
    # TODO maybe require input to be sorted first to prevent
    # sorting inputs that are already sorted again
    #news_report = deque([40,90,300])
    #intro =       deque([60,200,400])
    #print("peak_times before",peak_times)
    peak_times_clean = list(dict.fromkeys([peak for peak in sorted(peak_times)]))
    #print("peak_times after",peak_times)
    #exit(1)
    
    if remove_repeats:
        if not max_repeat_seconds:
            raise ValueError("max_repeat_seconds is required for remove repeats")
        # remove repeating beeps
        peak_times_clean = consolidate_close_by(peak_times_clean,max_seconds=max_repeat_seconds)
    
    return peak_times_clean


def consolidate_close_by(news_reports,max_seconds):
    if len(news_reports) == 0:
        return news_reports
    if not is_unique_and_sorted(news_reports):
        raise ValueError("news report is not unique or sorted")
    new_ones=[]
    #non_repeating_index = None
    #repeat_count = 0
    #max_seconds = BEEP_PATTERN_REPEAT_SECONDS
    cur_first = None
    for i,cur_news_report in enumerate(news_reports):
        if i == 0:
            cur_first=cur_news_report
            #print("cur_news_report",cur_news_report)
            #print("cur_first",cur_first)
            #print("add current and reset")
            new_ones.append(cur_news_report)
        else:
            #print("cur_news_report",cur_news_report)
            #print("cur_first",cur_first)
            #print("cur_news_report - cur_first",cur_news_report - cur_first)
            if (cur_news_report - cur_first <= max_seconds): #seconds
                pass
                #repeat_count += 1
            else:
                #print("add current and reset")
                #repeat_count = 0
                cur_first=cur_news_report
                #non_repeating_index=i
                new_ones.append(cur_news_report)
            #print("---------------")    
    return new_ones            
        
# news_reports need to be unique         
def consolidate_intros(intros,news_reports,total_time):
    if not is_unique_and_sorted(intros):
        raise ValueError("intros is not unique or sorted")
    if not is_unique_and_sorted(news_reports):
        raise ValueError("news report is not unique or sorted")
    
    for ts in intros:
        if ts > total_time:
            raise ValueError(f"intro overflow, is greater than total time {total_time}")
        elif ts < 0:
            raise ValueError(f"intro is less than 0")

    consolidated_intros = []


    #no news report or intro
    if len(news_reports) == 0 or len(intros) == 0:
        #just return beginning
        return [0] if len(intros) == 0 else [intros[0]]
 
    #normalize
    if(len(intros) > 0):
        if(intros[0]<0):
            raise ValueError("intro cannot be negative")
    else: # no intros
         return []   

   
    intros=deque(intros)
    news_reports=deque(news_reports)
    
    arr2=[]
    
    # min 1 intro and 1 news report
    while news_reports:
        temp=[]

        news = news_reports.popleft()
  
        # Check if there are extra intros before the current news
        while intros and intros[0] < news:
            temp.append(intros.popleft())
            #intro=intros.popleft()
        arr2.append(temp)

    arr2.append(intros)

    for arr in arr2:
        if len(arr) == 0:
            continue
        consolidated_intros.append(arr[0])
        
    return consolidated_intros
    
# clean up first 10 minutes and last 10 seconds
# first intro should not happen after 10 minutes
# first 10 minutes should have at most 1 news report, and if it does
# that first news report should be cut off
# last news report can only happen within 10 seconds of the end
def news_intro_process_beginning_and_end(intros,news_reports,total_time):
    if not is_unique_and_sorted(intros):
        raise ValueError("intros is not unique or sorted")
    if not is_unique_and_sorted(news_reports):
        raise ValueError("news report is not unique or sorted")
    
    news_reports=news_reports.copy()
    news_already = None
    for i,news_report in enumerate(news_reports):
        if(i > 1):
            break
        if(news_reports[i] <= INTRO_CUT_OFF):
            if(news_already is not None):
                raise TimeSequenceError("cannot have more than one news report within 10 minutes")
            else:
                news_already = news_report
    if(len(intros)>0):
        first_intro = intros[0]
        if(intros[0]>INTRO_CUT_OFF):
            raise TimeSequenceError("first intro cannot be greater than 10 minutes")            
    if(len(intros)==0 or len(news_reports)==0):
        return [total_time]

    # chop the first news report if it is less than 10 minutes
    if(news_already is not None and news_already<first_intro):
        news_reports=news_reports[1:]
    #else:
    #    news_reports=news_reports
    
    #treat news report as happening at the end
    if(len(news_reports)==0):
        return [total_time]
    
    if(intros[-1] > total_time):
        raise ValueError(f"intro overflow, is greater than total time {total_time}")


    end_cut_off_seconds = 10

    # make it complete
    if(news_reports[-1] < intros[-1]):
        news_reports.append(total_time)
    
    if(news_reports[-1] < total_time-end_cut_off_seconds):
        raise TimeSequenceError(f"cannot end with news reports unless it is within 10 seconds of the end to prevent missing things")
    
    return news_reports
                 
def build_time_sequence(start_times,end_times):
    if not is_unique_and_sorted(start_times):
        raise ValueError("start_times is not unique or sorted")
    if not is_unique_and_sorted(end_times):
        raise ValueError("end_times is not unique or sorted")
    if(len(start_times) != len(end_times)):
        intros_debug=[seconds_to_time(seconds=t,include_decimals=True) for t in start_times]
        news_reports_debug=[seconds_to_time(seconds=t,include_decimals=True) for t in end_times]
        raise TimeSequenceError(f"start_times and end_times must be the same length, otherwise it is sign of time sequence error:\n"+ 
                                f"start_times {intros_debug}\n  end_times {news_reports_debug}")
    result =[]
    for i in range(len(start_times)):
        result.append([start_times[i],end_times[i]])
    return remove_start_equals_to_end(result)
                 
def pad_news_report(time_sequences,total_time,news_report_second_pad=6):
    result=[]
    for i in range(1,len(time_sequences)):
        prev_seq=time_sequences[i-1]
        cur_seq=time_sequences[i]
        #enough room to pad
        if cur_seq[0] - prev_seq[1] >= news_report_second_pad:
            result.append([prev_seq[0],prev_seq[1]+news_report_second_pad])
        else:
            result.append([prev_seq[0],cur_seq[0]])
    if len(time_sequences) > 0:
        result.append([time_sequences[-1][0],seq if (seq:=time_sequences[-1][1]+news_report_second_pad) <= total_time else total_time])    
    return result   
                
def remove_start_equals_to_end(time_sequences):
    return list(filter(lambda x: (x[0] != x[1]), time_sequences)) 
    
# main function
def process_timestamps_rthk(news_reports,intros,total_time,news_report_second_pad=6,
                       allow_first_short=False):

    # if len(news_reports) != len(set(news_reports)):
    #    raise ValueError("news report has duplicates, clean up duplicates first")   

    # if len(intros) != len(set(intros)):
    #    raise ValueError("intro has duplicates, clean up duplicates first")   


    news_reports = preprocess_ts(news_reports,remove_repeats=True,max_repeat_seconds=BEEP_PATTERN_REPEAT_SECONDS)
    intros = preprocess_ts(intros)
    

    for ts in intros:
        if ts > total_time:
            raise ValueError(f"intro overflow, is greater than total time {total_time}")
        elif ts < 0:
            raise ValueError(f"intro is less than 0")

    # remove repeating intros
    intros = consolidate_intros(intros,news_reports,total_time)
    # process beginning and end
    news_reports = news_intro_process_beginning_and_end(intros,news_reports,total_time)

    time_sequences=build_time_sequence(start_times=intros,end_times=news_reports)
    time_sequences=pad_news_report(time_sequences,news_report_second_pad=news_report_second_pad,total_time=total_time)

    timestamp_sanity_check_rthk(time_sequences,allow_first_short=allow_first_short,total_time=total_time)

    return time_sequences

# TODO: still need to write tests for this
# this will limit the end to total_time unlike the rthk one, which allows end of time sequence to be greater than total time
def process_timestamps_single_intro(intros,total_time):

    intros = preprocess_ts(intros,remove_repeats=True,max_repeat_seconds=20)
    

    for ts in intros:
        if ts > total_time:
            raise ValueError(f"intro overflow, is greater than total time {total_time}")
        elif ts < 0:
            raise ValueError(f"intro is less than 0")

    end_times = []
    for i in range(1,len(intros)):
        intro=intros[i]
        end_times.append(intro)

    end_times.append(total_time)    

    time_sequences=build_time_sequence(start_times=intros,end_times=end_times)

    timestamp_sanity_check(time_sequences,total_time=total_time)

    return time_sequences