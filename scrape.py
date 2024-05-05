
import argparse
import copy
import glob
import sys
from collections import deque
import datetime
import hashlib
import json
import logging
import math
import os
import re
import shutil
import string
import subprocess
import tempfile
import traceback
from pathlib import Path

import ffmpeg
import paramiko
import pytz
import requests

from audio_offset_finder_v2 import convert_audio_to_clip_format, find_clip_in_audio_in_chunks, DEFAULT_METHOD, \
    cleanup_peak_times
from publish import publish_folder
from time_sequence_error import TimeSequenceError
from file_upload.upload_utils import upload_file
import utils
logger = logging.getLogger(__name__)

from andrew_utils import seconds_to_time
from utils import extract_prefix

streams={
    "happydaily": {
        "introclips": ["rthk1theme.wav","happydailyfirstintro.wav","happydailyfemaleintro.wav","happydailyfemale2.wav"],
        "allow_first_short": True,
        "url": "https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/happydaily/m4a/{date}.m4a/index_0_a.m3u8",
        "schedule":{"begin": 10,"end":12,"weekdays_human":[1,2,3,4,5]},
    },
    "healthpedia": {
        "introclips": ["rthk1theme.wav","healthpedia_intro.wav","healthpediapriceless.wav"],
        "allow_first_short": False,
        "url": "https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/healthpedia/m4a/{date}.m4a/index_0_a.m3u8",
        "schedule":{"begin": 13,"end":15,"weekdays_human":[1,2,3,4,5]},
    },
    "morningsuite": {
        "introclips": ["morningsuitethemefemalevoice.wav","morningsuitethememalevoice.wav","rthk2theme.wav"],
        "allow_first_short": False,
        "url":"https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio2/morningsuite/m4a/{date}.m4a/index_0_a.m3u8",
        "schedule":{"begin": 6,"end":10,"weekdays_human":[1,2,3,4,5]},
    },
    "KnowledgeCo": {
        "introclips": ["rthk2theme.wav","knowledgecointro.wav"],
        "allow_first_short": False,
        "url":"https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio2/KnowledgeCo/m4a/{date}.m4a/index_0_a.m3u8",
        "schedule":{"begin": 6,"end":8,"weekdays_human":[6]},
    },
}

# intro is almost always prominent
correlation_threshold_intro = 0.4
# news report is not always prominent
# especially with the longer beep2
correlation_threshold_news_report = 0.3

# use beep2 instead to reduce false positives, might
# live stream whole programs instead for easier processing
# with another unique news report clip
news_report_clip='rthk_beep2.wav'

news_report_black_list_ts = {
    "morningsuite20240424":[5342], # fake one
    "morningsuite20240502":[12538], # causing trouble
    #"KnowledgeCo20240427":[4157], # false positive 01:09:17
}

def url_ok(url):
 
 
    r = requests.get(url, stream=True)

    if r.ok:
        #content = next(r.iter_content(10))
        return True
    else:
        logger.error(f"HTTP Error {r.status_code} - {r.reason}")
        return False

def download(url,target_file):
    if(os.path.exists(target_file)):
        logger.info(f"file {target_file} already exists,skipping")
        return
    print(f'downloading {url}')
    with tempfile.TemporaryDirectory() as tmpdir:
        basename,extension = os.path.splitext(os.path.basename(target_file))
    
        tmp_file = os.path.join(tmpdir,f"download{extension}")
        (
        ffmpeg.input(url).output(tmp_file, **{'bsf:a': 'aac_adtstoasc'}, c='copy', loglevel="error")
              .run()
        )
        shutil.move(tmp_file,target_file)
    print(f'downloaded to {target_file}')

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

# total_time is needed to set end time
# if it ends with intro instead of news report
# did a count down and the beep intro for news report is about 6 seconds
# skip_reasonable_time_sequence_check: skip sanity checks related to unreasonable duration or gaps, mainly for testing
# otherwise will have to rewrite lots of tests if the parameters changed
# allow_first_short: allow first intro short
def process_timestamps(news_report,intro,total_time,news_report_second_pad=6,
                       allow_first_short=False):

    skip_reasonable_time_sequence_check=False

    # defensive copy
    news_report = copy.deepcopy(news_report)
    intro = copy.deepcopy(intro)
    

    if len(news_report) != len(set(news_report)):
       raise ValueError("news report has duplicates, clean up duplicates first")   

    if len(intro) != len(set(intro)):
       raise ValueError("intro has duplicates, clean up duplicates first")   


    # will bug out if not sorted
    # TODO maybe require input to be sorted first to prevent
    # sorting inputs that are already sorted again
    #news_report = deque([40,90,300])
    #intro =       deque([60,200,400])
    news_report=sorted(news_report)
    intro=sorted(intro)

    for ts in intro:
        if ts > total_time:
            raise ValueError(f"intro overflow, is greater than total time {total_time}")

    cur_intro = 0

    #news_report = deque([598, 2398, 3958, 5758])
    #intro = deque([1056, 2661, 4463])
    # no news report
    if(len(news_report)==0):
        # no need to trim
        return [[cur_intro, total_time]]
    
    # news report within the first 1 minute and it is less than the first intro, change to 0
    if(len(intro) > 0 and (news_report[0] <= 1*60 and news_report[0] < intro[0])):
        news_report[0]=0

    news_report=deque(news_report)
    intro=deque(intro)

    # intro starts before news report,
    # shift cur_intro from 0 to the first intro
    # if it is less than 10 minutes,
    # it is very unlikely to miss a news report
    # within the first 10 minutes and at the same time
    # the program has already started before 10 minutes
    if(len(intro) > 0 and intro[0] <= 10*60 and intro[0] < news_report[0]):
        cur_intro = intro.popleft()

    if(cur_intro > total_time):
        raise ValueError("intro overflow, is greater than total time {total_time}")

    
    # # absorb fake news report beep within 10 minutes of intro except short intro and news report close to the end
    # for i,ts_intro in enumerate(intro):
    #     seconds_absorb=10*60
    #     # not absorbing the news reports followed by first intro
    #     # if allow_first_short is true, if it still have false positives
    #     # just let it fail
    #     if allow_first_short and i == 0:
    #         continue
    #     news_report2=deque()
    #     while len(news_report)>0:
    #         news = news_report.popleft()
    #         if(news <= ts_intro):
    #             news_report2.append(news)
    #         elif(news < ts_intro + seconds_absorb):
    #             # absorb fake news report beep
    #             continue
    #         else:
    #             news_report2.append(news)
    #             #break
    #     news_report = news_report2


    pair=[]

    news_report_followed_by_intro = True
    while(len(news_report)>0):
        if(not news_report_followed_by_intro):
           raise ValueError("cannot have news report followed by news report")
        news_report_followed_by_intro=False
        cur_news_report = news_report.popleft()
        if(cur_intro > total_time):
            raise ValueError(f"intro overflow, is greater than total time {total_time}")
        # absorb fake news report beep within 10 minutes of intro except allow short intro and news report close to the end
        if len(pair) == 0:
            pass # no absorption for first pair because it is error prone
        else:
            while len(intro) > 0 and len(news_report)>0 and cur_news_report <= cur_intro + 10*60 and cur_news_report < intro[0]:
                cur_news_report=news_report.popleft()
        pair.append([cur_intro, cur_news_report])
        # get first intro after news report while ignoring others after first
        while(len(intro)>0):
             cur_intro = intro.popleft()
             if cur_intro > cur_news_report:
                 # ends with intro but no news report
                 if len(news_report)==0:
                    pair.append([cur_intro, total_time])

                 if(len(news_report)>0 and cur_intro > news_report[0]):
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

def split_audio(input_file, output_file, start_time, end_time,total_time,artist,album,title):

    metadata_list = ["title={}".format(title), "artist={}".format(artist), "album={}".format(album), ]
    metadata_dict = {f"metadata:g:{i}": e for i, e in enumerate(metadata_list)}

    (
    ffmpeg.input(input_file, ss=seconds_to_time(seconds=start_time,include_decimals=False), to=seconds_to_time(seconds=end_time,include_decimals=False))
            .output(output_file,acodec='copy',vcodec='copy', loglevel="error", **metadata_dict).overwrite_output().run()
    )

def concatenate_audio(input_files, output_file,tmpdir):
    list_file = os.path.join(tmpdir, 'list.txt')
    with open(list_file,'w') as f:
        for item in input_files:
            file_name = item["file_path"]
            print(f"file {file_name}",file=f)

    artist="rthk"

    basename,extension = os.path.splitext(os.path.basename(output_file))

    album,date_str = extract_prefix(basename)
    title=basename

    # add artist, album and title metadata
    #metadata_list = ["title={}".format(title), "artist={}".format(artist), "album={}".format(album), ]
    #metadata_dict = {f"metadata:g:{i}": e for i, e in enumerate(metadata_list)}
    text = f""";FFMETADATA1
artist={artist}
album={album}
title={title}\n"""
    start_time = 0
    chapter = ""
    for i in range(len(input_files)):
        duration = input_files[i]["end_time"]-input_files[i]["start_time"]
        end_time=start_time+duration
        path1=seconds_to_time(seconds=start_time,include_decimals=False).replace(':','_')
        path2=seconds_to_time(seconds=end_time,include_decimals=False).replace(':','_')
        title=f"{path1}-{path2}"
        text += f""";FFMETADATA1
[CHAPTER]
TIMEBASE=1/1
START={start_time}
END={end_time}
title={title}\n"""
        start_time = end_time

    ffmetadatafile = os.path.join(tmpdir, 'ffmetadatafile.txt')
    with open(ffmetadatafile, "w") as myfile:
        myfile.write(text)

    subprocess.run([
        'ffmpeg', 
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-f', 'ffmetadata',
        '-i', ffmetadatafile,
        '-map_metadata', '1',
        '-codec', 'copy',
        '-loglevel', 'error',
        '-y',
        output_file
    ])


def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def scrape(input_file,stream_name):

    print(input_file)
    #exit(1)
    basename,extension = os.path.splitext(os.path.basename(input_file))
    logger.info(basename)
    #md5=md5file(input_file)  # to get a printable str instead of bytes

    tsformatted = None

    jsonfile = f'{input_file}.json'
    if os.path.exists(jsonfile):
        tsformatted=json.load(open(jsonfile))['tsformatted']

    total_time = math.ceil(float(ffmpeg.probe(input_file)["format"]["duration"]))
    logger.debug("total_time",total_time,"---")
    #exit(1)
    if not tsformatted:
        stream = streams[stream_name]
        clips = stream["introclips"]
        allow_first_short = stream["allow_first_short"]

        # Find clip occurrences in the full audio
        news_report_peak_times = find_clip_in_audio_in_chunks(f'./audio_clips/{news_report_clip}',
                                                              input_file,
                                                              method=DEFAULT_METHOD,
                                                              correlation_threshold = correlation_threshold_news_report,
                                                              )
        news_report_peak_times = cleanup_peak_times(news_report_peak_times)
        audio_name,_ = os.path.splitext(os.path.basename(input_file))
        exclude_ts = news_report_black_list_ts.get(audio_name,None)
        if exclude_ts:
            news_report_peak_times = [time for time in news_report_peak_times if time not in exclude_ts]
            
        news_report_peak_times_formatted=[seconds_to_time(seconds=t,include_decimals=False) for t in news_report_peak_times]
        print("news_report_peak_times",news_report_peak_times_formatted,"---")
        for offset in news_report_peak_times:
            logger.info(
                f"Clip news_report_peak_times at the following times (in seconds): {seconds_to_time(seconds=offset, include_decimals=False)}")

        program_intro_peak_times=[]
        program_intro_peak_times_debug=[]
        for c in clips:
            #print(f"Finding {c}")
            intros=find_clip_in_audio_in_chunks(f'./audio_clips/{c}', input_file,method=DEFAULT_METHOD,correlation_threshold=correlation_threshold_intro)
            #print("intros",[seconds_to_time(seconds=t,include_decimals=False) for t in intros],"---")
            program_intro_peak_times.extend(intros)
            intros_debug = list(dict.fromkeys([math.floor(peak) for peak in intros]))
            program_intro_peak_times_debug.append({c:[intros_debug,[seconds_to_time(seconds=t,include_decimals=False) for t in intros_debug]]})
        program_intro_peak_times = cleanup_peak_times(program_intro_peak_times)
        logger.debug(program_intro_peak_times)
        print("program_intro_peak_times",[seconds_to_time(seconds=t,include_decimals=False) for t in program_intro_peak_times],"---")

        for offset in program_intro_peak_times:
            logger.info(f"Clip program_intro_peak_times at the following times (in seconds): {seconds_to_time(seconds=offset,include_decimals=False)}" )

        with open(f'{input_file}.separated.json','w') as f:
            f.write(json.dumps({"news_report":[news_report_peak_times,news_report_peak_times_formatted],"intros": program_intro_peak_times_debug}, indent=4))

        pair = process_timestamps(news_report_peak_times, program_intro_peak_times,total_time,allow_first_short=allow_first_short)
        #print("pair",pair)
        tsformatted = [[seconds_to_time(seconds=t,include_decimals=False) for t in sublist] for sublist in pair]

    else:
        pair = [[get_sec(t) for t in sublist] for sublist in tsformatted]
    #print(pair)
    #logger.debug("tsformatted",tsformatted)
    duration = [seconds_to_time(t[1]-t[0]) for t in pair]
    gaps=[]
    for i in range(1,len(pair)):
        gaps.append(seconds_to_time(pair[i][0]-pair[i-1][1]))
    with open(jsonfile,'w') as f:
        f.write(json.dumps({"tsformatted": tsformatted,"ts":pair,"duration":duration,"gaps":gaps}, indent=4))

    splits=[]

    output_dir_trimmed= os.path.abspath(os.path.join(f"./tmp","trimmed",stream_name))
    output_file_trimmed= os.path.join(output_dir_trimmed,f"{basename}_trimmed{extension}")

    # if os.path.exists(output_file):
    #     print(f"file {output_file} already exists,skipping")
    #     return
    
    os.makedirs(output_dir_trimmed, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename_trimmed=os.path.basename(output_file_trimmed)
        dirname,date_str = extract_prefix(filename_trimmed)
        dirname = '' if dirname is None else dirname
        for i,p in enumerate(pair):
            start_time = p[0]
            end_time = p[1]
            path1=seconds_to_time(seconds=start_time,include_decimals=False).replace(':','_')
            path2=seconds_to_time(seconds=end_time,include_decimals=False).replace(':','_')
            title=f"{path1}-{path2}"
            filename=f"{title}{extension}"
            file_segment = os.path.join(tmpdir,filename)
            split_audio(input_file, file_segment, start_time, end_time, total_time,artist=dirname,album=date_str,title=title)
            splits.append({"file_path": file_segment,
                           "start_time": start_time,
                           "end_time": end_time,})
        concatenate_audio(splits, output_file_trimmed,tmpdir)
        upload_path_trimmed = f"/rthk/trimmed/{dirname}/{filename_trimmed}"
        upload_file(output_file_trimmed,upload_path_trimmed,skip_if_exists=True)
        # save segments
        for item in splits:
            dirname_segment = os.path.abspath(f"./tmp/segments/{dirname}/{date_str}")
            os.makedirs(dirname_segment, exist_ok=True)
            filename_segment=os.path.basename(item["file_path"])
            save_path=f"{dirname_segment}/{filename_segment}"
            shutil.move(item["file_path"],save_path)
            #upload_file(item["file_path"],upload_path,skip_if_exists=True)
    return output_dir_trimmed,output_file_trimmed

def is_time_after(current_time,hour):
  target_time = datetime.time(hour, 0, 0)  # Set minutes and seconds to 0
  return current_time > target_time

def download_and_scrape(download_only=False):
    days_to_keep=14
    if days_to_keep < 1:
        raise ValueError("days_to_keep must be greater than or equal to 1")
    for key, stream in streams.items():
        error_occurred_scraping = False
        podcasts_publish=[] # should only be one per stream
        for days_ago in range(days_to_keep):
            date = datetime.datetime.now(pytz.timezone('Asia/Hong_Kong'))- datetime.timedelta(days=days_ago)
            date_str=date.strftime("%Y%m%d")
            url_template = stream['url']
            url = url_template.format(date=date_str)
            #print(key)
            schedule=stream['schedule']
            end_time_hour = schedule["end"]
            weekdays_human = schedule["weekdays_human"]
            if date.weekday()+1 not in weekdays_human:
                logger.info(f"skipping {key} because it is not scheduled for today's weekday")
                continue
            if days_ago == 0 and not is_time_after(date.time(),end_time_hour+1):
                logger.info(f"skipping {key} because it is not yet from {end_time_hour} + 1 hour")
                continue
            elif not url_ok(url):
                logger.warning(f"skipping {key} because url {url} is not ok")
                print(f"skipping {key} because url {url} is not ok")
                continue
            original_dir = os.path.abspath(f"./tmp/original/{key}")
            dest_file = os.path.join(original_dir,f"{key}{date_str}.m4a")
            os.makedirs(original_dir, exist_ok=True)
            try:
                download(url,dest_file)
                upload_file(dest_file,f"/rthk/original/{key}/{os.path.basename(dest_file)}",skip_if_exists=True)
                if(download_only):
                    continue
                output_dir_trimmed,output_file_trimmed = scrape(dest_file,stream_name=key)
                podcasts_publish.append(output_dir_trimmed)
            except Exception as e:
                print(f"error happened when processing for {key}",e)
                print(traceback.format_exc())
                error_occurred_scraping = True
                continue
        if error_occurred_scraping:
            print(f"error happened when processing, skipping publishing podcasts")
        elif not download_only:
            num_to_publish=3
            podcasts_publish = list(dict.fromkeys(podcasts_publish))
            for podcast in podcasts_publish:
                print(f"publishing podcast {podcast} after scraping")
                # assuming one per day
                publish_folder(podcast,files_to_publish=num_to_publish,delete_old_files=False)
            m4a_files_all = sorted(glob.glob(os.path.join(original_dir, "*.m4a")))
            # only keep last days_to_keep number of files, 
            # TODO: should account for weekends
            n = len(m4a_files_all) - days_to_keep
            n = 0 if n < 0 else n
            files_excluded = m4a_files_all[:n]
            #print(files_excluded)
            for file in files_excluded:
                print(f"deleting {file} and its jsons")
                Path(file).unlink(missing_ok=True)
                Path(f"{file}.json").unlink(missing_ok=True)
                Path(f"{file}.separated.json").unlink(missing_ok=True)
def command():
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')
    parser.add_argument('--pattern-file', metavar='audio file', type=str, help='pattern file to convert sample')
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()
    if(args.action == 'scrape'):
        input_file = args.audio_file
        stream_name = extract_prefix(os.path.split(input_file)[-1])[0]
        scrape(input_file,stream_name=stream_name)
    elif(args.action == 'convert'):
        input_file = args.pattern_file
        convert_audio_to_clip_format(input_file,os.path.splitext(input_file)[0]+"_converted.wav")
    elif(args.action == 'download'):
        download_and_scrape(download_only=True)
    elif(args.action == 'download_and_scrape'):
        download_and_scrape()
    else:
        raise ValueError(f"unknown action {args.action}")


    
if __name__ == '__main__':
    #print(url_ok("https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/happydaily/m4a/20240417.m4a/index_0_a.m3u8"))
    
    #upload_file("./tmp/out.pcm","/test5/5.pcm",skip_if_exists=True)
    
    #exit(1)
    #pair=[]
    #process(pair)
    #print(pair)
    command()
