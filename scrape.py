
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

from audio_offset_finder_v2 import convert_audio_to_clip_format, find_clip_in_audio_in_chunks, DEFAULT_METHOD
from process_timestamps import preprocess_ts, process_timestamps
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
        "introclips": ["rthk2theme.wav","knowledgecointro.wav","knowledge_co_e_word_intro.wav"],
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

# no need because it is absorbing now
news_report_black_list_ts = {
    "morningsuite20240424":[5342], # fake one 1 hr 29 min 2 sec
    "morningsuite20240502":[12538], # 3 hrs 28 min 58 sec causing trouble
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
    return int(h) * 3600 + int(m) * 60 + float(s)

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
        audio_name,_ = os.path.splitext(os.path.basename(input_file))
        exclude_ts = news_report_black_list_ts.get(audio_name,None)
        if exclude_ts:
            news_report_peak_times_filtered = []
            for second in preprocess_ts(news_report_peak_times,remove_repeats=True):
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

        program_intro_peak_times=[]
        program_intro_peak_times_debug=[]
        for c in clips:
            #print(f"Finding {c}")
            intros=find_clip_in_audio_in_chunks(f'./audio_clips/{c}', input_file,method=DEFAULT_METHOD,correlation_threshold=correlation_threshold_intro)
            #print("intros",[seconds_to_time(seconds=t,include_decimals=False) for t in intros],"---")
            program_intro_peak_times.extend(intros)
            intros_debug = sorted(intros)
            program_intro_peak_times_debug.append({c:[intros_debug,[seconds_to_time(seconds=t,include_decimals=True) for t in intros_debug]]})
        #program_intro_peak_times = cleanup_peak_times(program_intro_peak_times)
        #logger.debug(program_intro_peak_times)
        print("program_intro_peak_times",[seconds_to_time(seconds=t,include_decimals=True) for t in sorted(program_intro_peak_times)],"---")

        #for offset in program_intro_peak_times:
        #    logger.info(f"Clip program_intro_peak_times at the following times (in seconds): {seconds_to_time(seconds=offset,include_decimals=True)}" )

        with open(f'{input_file}.separated.json','w') as f:
            f.write(json.dumps({"news_report":[sorted(news_report_peak_times),news_report_peak_times_formatted],"intros": program_intro_peak_times_debug}, indent=4))

        pair = process_timestamps(news_report_peak_times, program_intro_peak_times,total_time,allow_first_short=allow_first_short)
        #print("pair before rehydration",pair)
        tsformatted = [[seconds_to_time(seconds=t,include_decimals=True) for t in sublist] for sublist in pair]

    else:
        pair = [[get_sec(t) for t in sublist] for sublist in tsformatted]
        #print("pair after rehydration",pair)
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
    failed_scrape_files=[]
    days_to_keep=19
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

    if failed_scrape_files:
        print(f"failed to scrape the following files:")
        for hash in failed_scrape_files:
            print("file",hash["file"],"error",hash["error"],"---")

def command():
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')
    parser.add_argument('--pattern-file', metavar='audio file', type=str, help='pattern file to convert sample')
    parser.add_argument('--dest-file', metavar='audio file', type=str, help='dest saved file')
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()
    if(args.action == 'scrape'):
        input_file = args.audio_file
        stream_name = extract_prefix(os.path.split(input_file)[-1])[0]
        scrape(input_file,stream_name=stream_name)
    elif(args.action == 'convert'):
        # python scrape.py convert --pattern-file  /Volumes/andrewdata/audio_test/knowledge_co_e_word_intro.wav --dest-file audio_clips/knowledge_co_e_word_intro.wav
        input_file = args.pattern_file
        convert_audio_to_clip_format(input_file,args.dest_file)
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
