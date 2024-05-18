
import argparse
import fcntl
import glob
from collections import deque
import datetime
import json
import logging
import math
import os
import shutil
import tempfile
import traceback
from pathlib import Path

import ffmpeg
import pytz

from audio_offset_finder_v2 import convert_audio_to_clip_format, find_clip_in_audio_in_chunks, DEFAULT_METHOD
#from database import save_debug_info_to_db
from process_timestamps import BEEP_PATTERN_REPEAT_SECONDS, preprocess_ts, process_timestamps_rthk
from publish import publish_folder
from scrape import concatenate_audio, download, get_sec, split_audio_by_time_sequences, url_ok
from time_sequence_error import TimeSequenceError
from file_upload.upload_utils2 import upload_file
logger = logging.getLogger(__name__)

from andrew_utils import seconds_to_time
from utils import extract_prefix

streams={
    "itsahappyday": {
        "introclips": ["itsahappyday_intro.wav"],
        "allow_first_short": True,
        "url": "https://rthkaod2022.akamaized.net/m4a/radio/archive/radio1/itsahappyday/m4a/{date}.m4a/master.m3u8",
        "schedule":{"end":12,"weekdays_human":[1,2,3,4,5]},
    },
    "healthpedia": {
        "introclips": ["healthpedia_intro.wav","healthpediapriceless.wav"],
        "allow_first_short": False,
        "url": "https://rthkaod2022.akamaized.net/m4a/radio/archive/radio1/healthpedia/m4a/{date}.m4a/master.m3u8",
        "schedule":{"end":15,"weekdays_human":[1,2,3,4,5]},
    },
    "morningsuite": {
        "introclips": ["morningsuitethemefemalevoice.wav","morningsuitethememalevoice.wav","morningsuitebababa.wav","morningsuiteinterlude1.wav"],
        "allow_first_short": False,
        "url":"https://rthkaod2022.akamaized.net/m4a/radio/archive/radio2/morningsuite/m4a/{date}.m4a/master.m3u8",
        "schedule":{"end":10,"weekdays_human":[1,2,3,4,5]},
    },
    "KnowledgeCo": {
        "introclips": ["rthk2theme_new.wav","knowledgecointro.wav","knowledge_co_e_word_intro.wav"],
        "allow_first_short": False,
        "url":"https://rthkaod2022.akamaized.net/m4a/radio/archive/radio2/KnowledgeCo/m4a/{date}.m4a/master.m3u8",
        "schedule":{"end":8,"weekdays_human":[6]},
    },
}

# intro is almost always prominent
correlation_threshold_intro = 0.4
# news report is not always prominent
# especially with the longer beep2
correlation_threshold_news_report = 0.4

# use beep2 instead to reduce false positives, might
# live stream whole programs instead for easier processing
# with another unique news report clip
news_report_clip='rthk_beep2.wav'

# no need because it is absorbing now
news_report_blacklist_ts = {
    #"morningsuite20240424":[5342], # fake one 1 hr 29 min 2 sec
    #"morningsuite20240502":[12538], # 3 hrs 28 min 58 sec causing trouble
    "KnowledgeCo20240511":[6176], # missing intro after 01:42:56
}


def scrape(input_file,stream_name):
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
    if os.path.exists(jsonfile):
        with open(jsonfile,'r') as f:
            tsformatted=json.load(f)['tsformatted']
    else:
        stream = streams[stream_name]
        clips = stream["introclips"]
        allow_first_short = stream["allow_first_short"]

        news_report_clip_path=f'./audio_clips/{news_report_clip}'

        clip_paths=[f'./audio_clips/{clip}' for clip in clips]

        clip_paths=[news_report_clip_path,*clip_paths]

        # Find clip occurrences in the full audio
        all_clip_peak_times = find_clip_in_audio_in_chunks(clip_paths=clip_paths,
                                                           full_audio_path=input_file,
                                                           method=DEFAULT_METHOD,
                                                           correlation_threshold = correlation_threshold_news_report,
                                                           )
        
        program_intro_peak_times=[]
        program_intro_peak_times_debug=[]
        for c in clips:
            intros=all_clip_peak_times[f'./audio_clips/{c}']
            #print("intros",[seconds_to_time(seconds=t,include_decimals=False) for t in intros],"---")
            program_intro_peak_times.extend(intros)
            intros_debug = sorted(intros)
            program_intro_peak_times_debug.append({c:[intros_debug,[seconds_to_time(seconds=t,include_decimals=True) for t in intros_debug]]})

        print("program_intro_peak_times",[seconds_to_time(seconds=t,include_decimals=True) for t in sorted(program_intro_peak_times)],"---")


        news_report_peak_times = all_clip_peak_times[news_report_clip_path]
        audio_name,_ = os.path.splitext(os.path.basename(input_file))
        exclude_ts = news_report_blacklist_ts.get(audio_name,None)
        if exclude_ts:
            news_report_peak_times_filtered = []
            for second in preprocess_ts(news_report_peak_times,remove_repeats=True,max_repeat_seconds=BEEP_PATTERN_REPEAT_SECONDS):
                #print(second)
                if math.floor(second) not in exclude_ts:
                    news_report_peak_times_filtered.append(second)
                else:
                    print(f"excluding {seconds_to_time(second)}, ({second}) seconds mark from news_report_peak_times")
            news_report_peak_times = news_report_peak_times_filtered        
        #exit(1)    
        news_report_peak_times_formatted=[seconds_to_time(seconds=t,include_decimals=True) for t in sorted(news_report_peak_times)]
        print("news_report_peak_times",news_report_peak_times_formatted,"---")
        #for offset in news_report_peak_times:
        #    logger.info(
        #        f"Clip news_report_peak_times at the following times (in seconds): {seconds_to_time(seconds=offset, include_decimals=False)}")

        #with open(f'{input_file}.separated.json','w') as f:
        #    f.write(json.dumps({"news_report":[sorted(news_report_peak_times),news_report_peak_times_formatted],"intros": program_intro_peak_times_debug}, indent=4))
        #save_debug_info_to_db(show_name,date_str,{"news_report":[sorted(news_report_peak_times),news_report_peak_times_formatted],"intros": program_intro_peak_times_debug})

        pair = process_timestamps_rthk(news_report_peak_times, program_intro_peak_times,total_time,allow_first_short=allow_first_short)
        #print("pair before rehydration",pair)
        tsformatted = [[seconds_to_time(seconds=t,include_decimals=True) for t in sublist] for sublist in pair]
        duration = [seconds_to_time(t[1]-t[0]) for t in pair]
        gaps=[]
        for i in range(1,len(pair)):
            gaps.append(seconds_to_time(pair[i][0]-pair[i-1][1]))
        with open(jsonfile,'w') as f:
            #print("jsonfile",jsonfile)
            content = json.dumps({"tsformatted": tsformatted,"ts":pair,"duration":duration,"gaps":gaps}, indent=4)
            #print(content)
            f.write(content)

    pair = [[get_sec(t) for t in sublist] for sublist in tsformatted]
    
    upload_file(jsonfile,f"/rthk/original/{show_name}/{os.path.basename(input_file)}.json",skip_if_exists=True)
    
    #save_timestamps_to_db(show_name,date_str,segments=tsformatted)

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
        scrape(input_file,stream_name=stream_name)
    elif(args.action == 'download'):
        download_and_scrape(download_only=True)
    elif(args.action == 'download_and_scrape'):
        download_and_scrape()
    else:
        raise ValueError(f"unknown action {args.action}")
