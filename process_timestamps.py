from collections import deque
import copy
import logging
import math
import sys

from time_sequence_error import TimeSequenceError
from andrew_utils import seconds_to_time
from utils import find_nearest_distance_forward, is_unique_and_sorted

logger = logging.getLogger(__name__)

# allow one intro and news report within 10 minutes
# but not intro past 10 minutes
INTRO_CUT_OFF=10*60

def timestamp_sanity_check(result,total_time):

    if(len(result) == 0):
        raise ValueError("result cannot be empty")

    for i,r in enumerate(result):
        if(len(r) != 2):
            raise ValueError(f"each element in result must have 2 elements, got {r}")

        #beginning = i == 0
        # not allow end to be larger than total time for now
        #end = i == len(result)-1

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
   
    
    if(result[-1][1] > total_time):
        raise TimeSequenceError(f"no longer allow end time {result[-1][1]} greater than total time {total_time}")

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
def consolidate_intros(intros,news_reports,total_time,backup_intro_ts=[]):
    if not is_unique_and_sorted(intros):
        raise ValueError("intros is not unique or sorted")
    if not is_unique_and_sorted(news_reports):
        raise ValueError("news report is not unique or sorted")
    
    for ts in intros:
        if ts > total_time:
            raise TimeSequenceError(f"intro overflow, is greater than total time {total_time}")
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
        
    if len(consolidated_intros) > 0 and consolidated_intros[0] > INTRO_CUT_OFF and len(backup_intro_ts) > 0:
        # need to pad an earlier intro from backup if any
        closest_backup_intro_ts = min(backup_intro_ts)
        if(closest_backup_intro_ts <= INTRO_CUT_OFF):
            consolidated_intros[0]=closest_backup_intro_ts

    return consolidated_intros
    
# clean up first 10 minutes 
# first intro should not happen after 10 minutes
# first 10 minutes should have at most 1 news report, and if it does
# that first news report should be cut off
# last news report can only happen within 2 minutes of the end
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
        raise TimeSequenceError(f"intro overflow, is greater than total time {total_time}")


    end_cut_off_seconds = 2*60

    # make it complete
    if(news_reports[-1] < intros[-1]):
        news_reports.append(total_time)
    
    if(news_reports[-1] < total_time-end_cut_off_seconds):
        raise TimeSequenceError(f"cannot end with news reports unless it is within 2 minutes of the end to prevent missing things")
    
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

# to be called before build_time_sequence where all starts and ends are included
def absorb_fake_news_report(intros,new_reports):
    if not is_unique_and_sorted(intros):
        raise ValueError("start_times is not unique or sorted")
    if not is_unique_and_sorted(new_reports):
        raise ValueError("end_times is not unique or sorted")
    
    # don't try
    if(len(intros) >= len(new_reports)):
        return new_reports.copy()
    
    exclude_list=[]

    for i,intro in enumerate(intros):
        if(new_reports[i] - intro > 0 and new_reports[i] - intro < 10*60): # only consider less than 10 minutes
            #excluded_one = False
            if(new_reports[i+1]-intro < 26*60): # don't exclude if excluding it will make it longer than 26 minutes
                # still won't work if there is legit long sequence or a real one was cut off but the fake one happens early
                exclude_list.append(i)
                #excluded_one = True
            #if excluded_one and new_reports[i+1] - new_reports[i] < 60:
            #    # exclude one more close by
            #    exclude_list.append(i+1)

    return [news_report for i, news_report in enumerate(new_reports) if i not in exclude_list]            
    
# not really need to sort backup_intro_ts    
def pad_from_backup_intro_ts(intros,backup_intro_ts,news_reports):
    if not is_unique_and_sorted(intros):
        raise ValueError("start_times is not unique or sorted")
    if not is_unique_and_sorted(news_reports):
        raise ValueError("end_times is not unique or sorted")
    # defensive copy
    intros = intros.copy()
    if len(news_reports) == 0:
        return intros
    if len(backup_intro_ts) == 0:
        return intros
    if len(intros) == 0:
        return [min(backup_intro_ts)]

    if len(intros) < len(news_reports) and len(backup_intro_ts) > 0:
        intros_new = []
        intros = deque(intros)
        placeholder = None
        intros.append(placeholder)
        i=0
        while len(intros) > 0 and len(intros_new) + len(intros) <= len(news_reports):
            appended = False
            if intros[0] is None or intros[0] > news_reports[i]:
                prev_news_report = 0 if i == 0 else news_reports[i-1]
                closest_backup_intro_dist = find_nearest_distance_forward(backup_intro_ts, prev_news_report)
                if(closest_backup_intro_dist is None):
                    logger.warning(f"find_nearest_distance_forward returned None for {prev_news_report}")        
                elif(closest_backup_intro_dist > 0 and closest_backup_intro_dist <= 10*60):
                    closest_backup_intro_ts = prev_news_report + closest_backup_intro_dist
                    if len(intros_new) > 0 and closest_backup_intro_ts <= intros_new[-1]:
                        logger.warning(f"closest_backup_intro_ts {seconds_to_time(closest_backup_intro_ts)} >= {seconds_to_time(intros_new[-1])}, ignoring")
                    elif closest_backup_intro_ts < news_reports[i]:
                        print(f"inserting backup intro at {seconds_to_time(closest_backup_intro_ts)}")
                        intros.appendleft(closest_backup_intro_ts)
                        appended = True
                else:
                    logger.warning(f"closest_backup_intro_dist {closest_backup_intro_dist} is farther than 10 minutes from {prev_news_report}")       
                if not appended:
                    logger.warning(f"no backup intro found to be appendable for {intros[0]}")        
            if appended:
                continue
            else:
                cur = intros.popleft()
                if cur != placeholder:
                    intros_new.append(cur)
                i+=1
        while len(intros) > 0:
            cur = intros.popleft()
            if cur != placeholder:
                intros_new.append(cur)
        if not is_unique_and_sorted(intros_new):
            raise ValueError(f"intros_new afterwards {[seconds_to_time(i) for i in intros_new]} is not unique or sorted")
        return intros_new    
    else:
        return intros    
    
# only works if news report timestamp is accurate    
def fill_in_short_intervals_missing_intros(intros,news_reports):
    if not is_unique_and_sorted(intros):
        raise ValueError(f"start_times {[seconds_to_time(i) for i in intros]} is not unique or sorted")
    if not is_unique_and_sorted(news_reports):
        raise ValueError("end_times is not unique or sorted")
    # defensive copy
    intros = intros.copy()
    if len(news_reports) == 0:
        return intros
    if len(intros) == 0:
        return intros
    if len(intros) < len(news_reports):
        intros_new = []
        intros = deque(intros)
        placeholder = None
        intros.append(placeholder)
        #print("intros",intros)
        #exit(1)
        i=0
        while len(intros) > 0 and len(intros_new) + len(intros) <= len(news_reports):
            appended = False
            if intros[0] is None or intros[0] > news_reports[i]:
                prev_news_report = 0 if i == 0 else news_reports[i-1]
                if(news_reports[i] - prev_news_report < 24*60): #make up if less than 24 minutes
                    print(f"adding news report at {seconds_to_time(prev_news_report)} before {seconds_to_time(intros[0]) if intros[0] is not None else 'end'} because clip is short and nothing is found")
                    intros.appendleft(prev_news_report)
                    appended = True
                else:
                    logger.warning(f"news_reports[i] {seconds_to_time(news_reports[i])} is farther than 24 minutes from {prev_news_report}")       
                if not appended:
                    logger.warning(f"no backup intro found to be appendable for {seconds_to_time(intros[0]) if intros[0] is not None else 'end'}")        
            if appended:
                continue
            else:
                cur = intros.popleft()
                if cur != placeholder:
                    intros_new.append(cur)
                i+=1
        while len(intros) > 0:
            cur = intros.popleft()
            if cur != placeholder:
                intros_new.append(cur)
        if not is_unique_and_sorted(intros_new):
            raise ValueError(f"intros_new afterwards {[seconds_to_time(i) for i in intros]} is not unique or sorted")
        return intros_new    
    else:
        return intros
                 
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
    
# main function, if news_report_strategy is theme_clip, won't absorb fake news report
def process_timestamps_rthk(news_reports,intros,total_time,news_report_second_pad=0,backup_intro_ts=[],
                       allow_first_short=False,news_report_strategy="beep"):

    # if len(news_reports) != len(set(news_reports)):
    #    raise ValueError("news report has duplicates, clean up duplicates first")   

    # if len(intros) != len(set(intros)):
    #    raise ValueError("intro has duplicates, clean up duplicates first")   


    news_reports = preprocess_ts(news_reports)
    intros = preprocess_ts(intros)
    

    for ts in intros:
        if ts > total_time:
            raise TimeSequenceError(f"intro overflow, is greater than total time {total_time}")
        elif ts < 0:
            raise ValueError(f"intro is less than 0")

    # remove repeating intros and add backup one at beginning if needed
    intros = consolidate_intros(intros,news_reports,total_time,backup_intro_ts=backup_intro_ts)
    # process beginning and end
    news_reports = news_intro_process_beginning_and_end(intros,news_reports,total_time)

    if news_report_strategy == "beep":
        # absorb fake news report before building time sequence
        news_reports = absorb_fake_news_report(intros,news_reports)
    
    #print("intros",intros,"news_reports",news_reports,"backup_intro_ts",backup_intro_ts)
    intros = pad_from_backup_intro_ts(intros,backup_intro_ts=backup_intro_ts,news_reports=news_reports)

    if news_report_strategy == "theme_clip":
        #pass
        intros = fill_in_short_intervals_missing_intros(intros,news_reports=news_reports)

    time_sequences=build_time_sequence(start_times=intros,end_times=news_reports)
    time_sequences=pad_news_report(time_sequences,news_report_second_pad=news_report_second_pad,total_time=total_time)

    timestamp_sanity_check_rthk(time_sequences,allow_first_short=allow_first_short,total_time=total_time)

    return time_sequences

def process_timestamps_simple(intros,endings,total_time,expected_num_segments=None,ends_with_intro=False,intro_max_repeat_seconds=None):
    marker_intro=0
    marker_ending=1
    if intro_max_repeat_seconds:
        intros = preprocess_ts(intros,remove_repeats=True,max_repeat_seconds=intro_max_repeat_seconds)
    else:
        intros = preprocess_ts(intros,remove_repeats=False)
    endings = preprocess_ts(endings,remove_repeats=False)
    

    #print("intros before",intros)
    #print("endings before",endings)       

    if(ends_with_intro):
        if(len(intros)<2):
            ends_with_intro = False
            #raise ValueError("Not enough intros found for ends_with_intro")
    if(ends_with_intro):    
        intro_ending = intros[-1]
        #print("len(endings)",len(endings))
        #print("endings[-1]",endings[-1])
        #print("intro_ending",intro_ending)
        if len(endings) > 0 and endings[-1] > intro_ending:
            ends_with_intro = False
            #raise ValueError("ends_with_intro is set but ending is greater than last intro")
        else:
            endings.append(intro_ending)
            intros = intros[:-1]

   # print("ends_with_intro",ends_with_intro)

    if not ends_with_intro:
        if(len(endings)==0 or endings[-1] < total_time):   
            endings.append(total_time)    

    #print("intros before2",intros)
    #print("endings before2",endings)        

    if(len(endings)>0 and endings[-1] > total_time):
        raise TimeSequenceError(f"ending overflow, is greater than total time {total_time}")

    for ts in intros:
        if ts > total_time:
            raise TimeSequenceError(f"intro overflow, is greater than total time {total_time}")
        elif ts < 0:
            raise ValueError(f"intro is less than 0")
        
    if(len(intros) == 0):
        raise ValueError("intros cannot be empty")    

    arr=[]
    for intro in intros:
        arr.append((intro,marker_intro))

    for ending in endings:
        arr.append((ending,marker_ending))    

    arr.sort(key=lambda x: x[0])
    arr = deque(arr)

    start_times = []
    end_times = []
    
    #print("arr before",arr)

    prev_marker = None
    # pop all endings before the first intro
    while len(arr) > 0:
        item = arr[0]
        if item[1] == marker_intro:
            prev_marker = marker_intro
            start_times.append(item[0])
            arr.popleft()
            break
        else:
            prev_marker = marker_ending
            arr.popleft()
    
    #print(arr)
    #print(start_times)
    #print(end_times)

    while len(arr) > 0:
        item = arr.popleft()
        cur_marker = item[1]
        if prev_marker is None:
            raise ValueError("prev_marker should not be None")
        intro_followed_by_intro = prev_marker == marker_intro and cur_marker == marker_intro
        intro_followed_by_ending = prev_marker == marker_intro and cur_marker == marker_ending
        ending_followed_by_ending = prev_marker == marker_ending and cur_marker == marker_ending
        ending_followed_by_intro = prev_marker == marker_ending and cur_marker == marker_intro
        prev_marker = cur_marker
        if intro_followed_by_intro:
            start_times.append(item[0])
            end_times.append(item[0])
        elif intro_followed_by_ending:
            end_times.append(item[0]) 
        elif ending_followed_by_ending:
            pass
        elif ending_followed_by_intro:       
            start_times.append(item[0])
        else:
            raise ValueError(f"unexpected marker combination {prev_marker} and {cur_marker}")

    #ending = total_time if ending > total_time else ending    

    # end_times = []
    # for i in range(1,len(intros)):
    #     intro=intros[i]
    #     end_times.append(intro)


    #end_times.append(ending)

    time_sequences=build_time_sequence(start_times=start_times,end_times=end_times)

    #print([[seconds_to_time(seconds=t,include_decimals=True) for t in sublist] for sublist in time_sequences])

    if(expected_num_segments and len(time_sequences) != expected_num_segments):
        raise TimeSequenceError(f"expected {expected_num_segments} segments, got {len(time_sequences)} segments")

    timestamp_sanity_check(time_sequences,total_time=total_time)

    return time_sequences