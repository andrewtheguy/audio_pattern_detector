
import argparse
import fcntl
import glob
from collections import deque
import datetime
import json
import logging
import math
import os
import pprint
import shutil
import tempfile
import traceback
from pathlib import Path

import ffmpeg
import numpy as np
import pytz

from audio_offset_finder_v2 import cleanup_peak_times, convert_audio_to_clip_format, find_clip_in_audio_in_chunks, DEFAULT_METHOD
#from database import save_debug_info_to_db
from process_timestamps import preprocess_ts, process_timestamps_rthk
from publish import publish_folder
from scrape import concatenate_audio, download, get_sec, split_audio_by_time_sequences, url_ok
from time_sequence_error import TimeSequenceError
from file_upload.upload_utils2 import upload_file
logger = logging.getLogger(__name__)

from andrew_utils import seconds_to_time
from utils import extract_prefix, find_nearest_distance_backwards, get_ffprobe_info

streams={
    "itsahappyday": {
        "introclips": ["itsahappyday_intro.wav"],
        "allow_first_short": True,
        "url": "https://rthkaod2022.akamaized.net/m4a/radio/archive/radio1/itsahappyday/m4a/{date}.m4a/master.m3u8",
        "schedule":{"end":12,"weekdays_human":[1,2,3,4,5]},
    },
    "healthpedia": {
        "introclips": ["healthpedia_intro.wav","healthpediapriceless.wav","healthpediamiddleintro.wav"],
        "allow_first_short": False,
        "url": "https://rthkaod2022.akamaized.net/m4a/radio/archive/radio1/healthpedia/m4a/{date}.m4a/master.m3u8",
        "schedule":{"end":15,"weekdays_human":[1,2,3,4,5]},
    },
    # rthk2 needs a different strategy for news report because it is less consistent
    "morningsuite": {
        "introclips": ["morningsuitethemefemalevoice.wav","morningsuitethememalevoice.wav","morningsuitebababa.wav","morningsuiteinterlude1.wav"],
        "allow_first_short": False,
        "url":"https://rthkaod2022.akamaized.net/m4a/radio/archive/radio2/morningsuite/m4a/{date}.m4a/master.m3u8",
        "schedule":{"end":10,"weekdays_human":[1,2,3,4,5]},
        "news_report_strategy":"theme_clip",
        "news_report_strategy_expected_count":4,
    },
    "KnowledgeCo": {
        "introclips": ["rthk2theme.wav","rthk2theme_new.wav","knowledgecointro.wav","knowledge_co_e_word_intro.wav"],
        "allow_first_short": False,
        "url":"https://rthkaod2022.akamaized.net/m4a/radio/archive/radio2/KnowledgeCo/m4a/{date}.m4a/master.m3u8",
        "schedule":{"end":8,"weekdays_human":[6]},
        "news_report_strategy":"theme_clip",
        "news_report_strategy_expected_count":3,
    },
}

# intro is almost always prominent
correlation_threshold_intro = 0.4

def get_by_news_report_strategy_beep(input_file):
    news_report_blacklist_ts = {
        #"morningsuite20240424":[5342], # fake one 1 hr 29 min 2 sec
        #"morningsuite20240502":[12538], # 3 hrs 28 min 58 sec causing trouble
        "KnowledgeCo20240511":[6176], # missing intro after 01:42:56
    }

    beep_pattern_repeat_seconds = 7

    # use beep2 instead to reduce false positives, might
    # live stream whole programs instead for easier processing
    # with another unique news report clip
    news_report_clip='rthk_beep2.wav'
    news_report_clip_path=f'./audio_clips/{news_report_clip}'

    clip_paths_news_report=[news_report_clip_path]

    # news report is not always prominent
    # especially with the longer beep2
    correlation_threshold_news_report = 0.3
    news_report_clip_peak_times = find_clip_in_audio_in_chunks(clip_paths=clip_paths_news_report,
                                                        full_audio_path=input_file,
                                                        method=DEFAULT_METHOD,
                                                        correlation_threshold = correlation_threshold_news_report,
                                                        )

    news_report_peak_times = news_report_clip_peak_times[news_report_clip_path]
    audio_name,_ = os.path.splitext(os.path.basename(input_file))
    exclude_ts = news_report_blacklist_ts.get(audio_name,None)
    
    news_report_peak_times_filtered = []
    for second in preprocess_ts(news_report_peak_times,remove_repeats=True,max_repeat_seconds=beep_pattern_repeat_seconds):
        if exclude_ts:
            if math.floor(second) not in exclude_ts:
                news_report_peak_times_filtered.append(second)
            else:
                print(f"excluding {seconds_to_time(second)}, ({second}) seconds mark from news_report_peak_times")
        else:
            news_report_peak_times_filtered.append(second)        
        news_report_peak_times = news_report_peak_times_filtered   
    return news_report_peak_times 

# only used for helping get_by_news_report_theme_clip for now
# can tolerate some inaccuracies
def get_single_beep(input_file):

    # use beep2 instead to reduce false positives, might
    # live stream whole programs instead for easier processing
    # with another unique news report clip
    news_report_clip='rthk_beep.wav'
    news_report_clip_path=f'./audio_clips/{news_report_clip}'

    clip_length_second = float(get_ffprobe_info(news_report_clip_path)['format']['duration'])
    #pprint.pprint(metadata)
    #raise "chafa"

    clip_paths_news_report=[news_report_clip_path]

    # higher threshold because it is a short beep
    correlation_threshold_news_report = 0.6
    news_report_clip_peak_times = find_clip_in_audio_in_chunks(clip_paths=clip_paths_news_report,
                                                        full_audio_path=input_file,
                                                        method=DEFAULT_METHOD,
                                                        correlation_threshold = correlation_threshold_news_report,
                                                        )
    
    news_report_peak_times = news_report_clip_peak_times[news_report_clip_path]
    
    return news_report_peak_times,clip_length_second
    #return preprocess_ts(news_report_peak_times,remove_repeats=False)



# it is easier to match using the news report theme clip than beep
def get_by_news_report_theme_clip(input_file,news_report_strategy_expected_count,total_time):

    if news_report_strategy_expected_count < 1:
        raise ValueError("news_report_strategy_expected_count must be greater than or equal to 1")
    
    #second_backtrack = 8

    single_beep_ts,clip_length_second=get_single_beep(input_file)
    cleanup_single_beep_ts = [seconds_to_time(seconds=t,include_decimals=True) for t in cleanup_peak_times(single_beep_ts)]
    #print("cleanup_single_beep_ts single beep",cleanup_single_beep_ts,"---")
    #print("clip_length_second single beep",clip_length_second,"---")

    # use beep2 instead to reduce false positives, might
    # live stream whole programs instead for easier processing
    # with another unique news report clip
    #news_report_clip='rthk_news_report_theme.wav'
    news_report_clip_path=f'./audio_clips/rthk_news_report_theme.wav'

    clip_paths_news_report=[news_report_clip_path]

    correlation_threshold_news_report = 0.7
    news_report_clip_peak_times = find_clip_in_audio_in_chunks(clip_paths=clip_paths_news_report,
                                                        full_audio_path=input_file,
                                                        method=DEFAULT_METHOD,
                                                        correlation_threshold = correlation_threshold_news_report,
                                                        )

    news_report_peak_times = news_report_clip_peak_times[news_report_clip_path]

    # sort and remove dup
    news_report_peak_times = preprocess_ts(news_report_peak_times,remove_repeats=False)

    if len(news_report_peak_times) != news_report_strategy_expected_count:
        raise ValueError(f"expected {news_report_strategy_expected_count} news reports but found {len(news_report_peak_times)}")
    
    if(news_report_peak_times[0] > 30 * 60):
        raise ValueError("first news report is too late, should not happen unless there is really a valid case for it")
    
    for i in range(1,len(news_report_peak_times)):
        if news_report_peak_times[i] - news_report_peak_times[i-1] < 45*60:
            raise ValueError("distance between news reports is too short, should not happen unless there is really a valid case for it")
        elif news_report_peak_times[i] - news_report_peak_times[i-1] > 60*60:
            raise ValueError("distance between news reports is more than an hour in between, should not happen unless there is really a valid case for it")
        
    news_report_final = []
    #print("news_report_peak_times before",news_report_peak_times,"---")
    for i,second in enumerate(news_report_peak_times):
        if second > total_time:
            raise ValueError("news report theme cannot be after total time")
        
        second_backtrack = find_nearest_distance_backwards(single_beep_ts,second)-clip_length_second
        #print('second_backtrack',second_backtrack,'---')
        if second_backtrack > 10:
            # could be theme found but not beep
            print("warn: second_backtrack where theme happens too far from the beep, potentially a bug or just no beep happening in the middle, changing it to 8")
            second_backtrack=8
            #raise ValueError("news report theme is too far from the beep, potentially a bug")
        
        #print('second_backtrack',second_backtrack,'---')

        second_beg = second - second_backtrack

        news_report_final.append(second_beg)
        # add 30 minutes after each news report, then backtrack to the nearest beep
        next_report = second_beg+30*60
        next_report_second_backtrack = find_nearest_distance_backwards(single_beep_ts,next_report)-clip_length_second
        if next_report_second_backtrack > 10:
            # could be beep not prominent enough
            print("warn: next_report_second_backtrack too far from the beep, potentially a bug or just no beep happening in the middle, changing it to 0")
            next_report_second_backtrack=0
        #print('next_report_second_backtrack',second_backtrack,'---')    
        next_report = next_report - next_report_second_backtrack
        
        if next_report < total_time:
            news_report_final.append(next_report)
        #if i == len(news_report_peak_times)-1:
        #    pass
    #print("news_report_final",news_report_final,"---")
    return news_report_final

def scrape(input_file,stream_name,always_reprocess=False):
    save_segments = False
    print(input_file)
    #exit(1)
    basename,extension = os.path.splitext(os.path.basename(input_file))
    logger.info(basename)
    #md5=md5file(input_file)  # to get a printable str instead of bytes

    show_name,date_str = extract_prefix(basename)

    tsformatted = None


    total_time = math.ceil(float(ffmpeg.probe(input_file)["format"]["duration"]))
    logger.debug("total_time",total_time,"---")
    

    jsonfile = f'{input_file}.json'
    if (not always_reprocess) and os.path.exists(jsonfile):
        with open(jsonfile,'r') as f:
            tsformatted=json.load(f)['tsformatted']
    else:
        stream = streams[stream_name]
        clips = stream["introclips"]
        allow_first_short = stream["allow_first_short"]
        news_report_strategy=stream.get("news_report_strategy","beep")
        news_report_strategy_expected_count=stream.get("news_report_strategy_expected_count",None)



        if news_report_strategy == "beep":
            news_report_peak_times = get_by_news_report_strategy_beep(input_file=input_file)
            news_report_second_pad = 6
        elif news_report_strategy == "theme_clip":
            if news_report_strategy_expected_count is None:
                raise ValueError("news_report_strategy_expected_count must be set when strategy is theme_clip")
            news_report_peak_times = get_by_news_report_theme_clip(input_file=input_file,news_report_strategy_expected_count=news_report_strategy_expected_count,total_time=total_time)
            news_report_second_pad = 0
        else:
            raise ValueError(f"unknown news report strategy {news_report_strategy}")
        
        news_report_peak_times_formatted=[seconds_to_time(seconds=t,include_decimals=True) for t in sorted(news_report_peak_times)]
        print("news_report_peak_times",news_report_peak_times_formatted,"---")

        clip_paths_intros=[f'./audio_clips/{clip}' for clip in clips]

        # Find clip occurrences in the full audio
        intro_clip_peak_times = find_clip_in_audio_in_chunks(clip_paths=clip_paths_intros,
                                                           full_audio_path=input_file,
                                                           method=DEFAULT_METHOD,
                                                           correlation_threshold = correlation_threshold_intro,
                                                           )
        
        program_intro_peak_times=[]
        program_intro_peak_times_debug=[]
        for c in clips:
            intros=intro_clip_peak_times[f'./audio_clips/{c}']
            #print("intros",[seconds_to_time(seconds=t,include_decimals=False) for t in intros],"---")
            program_intro_peak_times.extend(intros)
            intros_debug = sorted(intros)
            program_intro_peak_times_debug.append({c:[intros_debug,[seconds_to_time(seconds=t,include_decimals=True) for t in intros_debug]]})

        print("program_intro_peak_times",[seconds_to_time(seconds=t,include_decimals=True) for t in sorted(program_intro_peak_times)],"---")

        pair = process_timestamps_rthk(news_report_peak_times, program_intro_peak_times,total_time,allow_first_short=allow_first_short,news_report_second_pad=news_report_second_pad)
        tsformatted = [[seconds_to_time(seconds=t,include_decimals=True) for t in sublist] for sublist in pair]
        print("final sequences",tsformatted)
        duration = [seconds_to_time(t[1]-t[0]) for t in pair]
        distance_endings=[]
        for i in range(1,len(pair)):
            distance_endings.append(seconds_to_time(pair[i][1]-pair[i-1][1]))
        with open(jsonfile,'w') as f:
            #print("jsonfile",jsonfile)
            content = json.dumps({"tsformatted": tsformatted,"ts":pair,"duration":duration,"distance_endings":distance_endings}, indent=4)
            #print(content)
            f.write(content)

    pair = [[get_sec(t) for t in sublist] for sublist in tsformatted]
    
    upload_file(jsonfile,f"/rthk/original/{show_name}/{os.path.basename(input_file)}.json",skip_if_exists=True)
    
    #save_timestamps_to_db(show_name,date_str,segments=tsformatted)

    output_dir_trimmed= os.path.abspath(os.path.join(f"./tmp","trimmed",stream_name))
    output_file_trimmed= os.path.join(output_dir_trimmed,f"{basename}_trimmed{extension}")

    os.makedirs(output_dir_trimmed, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename_trimmed=os.path.basename(output_file_trimmed)
        dirname,date_str = extract_prefix(filename_trimmed)
        dirname = '' if dirname is None else dirname
        splits=split_audio_by_time_sequences(input_file,total_time,pair,tmpdir)
        concatenate_audio(splits, output_file_trimmed,tmpdir,channel_name="rthk",total_time=total_time)
        upload_path_trimmed = f"/rthk/trimmed/{dirname}/{filename_trimmed}"
        upload_file(output_file_trimmed,upload_path_trimmed,skip_if_exists=True)
        if save_segments:
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
    try:
        lockfile = open(f'./tmp/lockfile', "a+")
        fcntl.lockf(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        raise RuntimeError('can only run one download_and_scrape at a time')
   
    failed_scrape_files=[]
    days_to_keep=3
    if days_to_keep < 1:
        raise ValueError("days_to_keep must be greater than or equal to 1")
    for key, stream in streams.items():
        error_occurred_scraping = False
        podcasts_publish=[] # should only be one per stream
        original_dir = os.path.abspath(f"./tmp/original/{key}")
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
                failed_scrape_files.append({"file":dest_file,"error":str(e)})
                continue
        if error_occurred_scraping:
            print(f"error happened when processing, skipping publishing podcasts")
        elif not download_only:
            num_to_publish=3
            podcasts_publish = list(dict.fromkeys(podcasts_publish))
            for podcast in podcasts_publish:
                print(f"publishing podcast {podcast} after scraping")
                # assuming one per day
                publish_folder(podcast,files_to_publish=num_to_publish,delete_old_files=True)
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
                #Path(f"{file}.separated.json").unlink(missing_ok=True)

    if failed_scrape_files:
        print(f"failed to scrape the following files:")
        for hash in failed_scrape_files:
            print("file",hash["file"],"error",hash["error"],"---")

if __name__ == '__main__':
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
        scrape(input_file,stream_name=stream_name,always_reprocess=True)
    elif(args.action == 'download'):
        download_and_scrape(download_only=True)
    elif(args.action == 'download_and_scrape'):
        download_and_scrape()
    else:
        raise ValueError(f"unknown action {args.action}")

