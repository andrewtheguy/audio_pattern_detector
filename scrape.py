
import argparse
from collections import deque
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import string
import tempfile
import traceback

import ffmpeg
import paramiko
import pytz
import requests

from audio_offset_finder_v2 import convert_audio_to_clip_format, find_clip_in_audio_in_chunks
from time_sequence_error import TimeSequenceError
from upload_utils import upload_file

introclips={
    "happydaily":["happydailyfirstintro.wav","happydailyfemaleintro.wav","happydailyfemale2.wav"],
    "healthpedia":["rthk1theme.wav","healthpedia_intro.wav"],
    "morningsuite":["morningsuitethemefemalevoice.wav","morningsuitethememalevoice.wav","rthk2theme.wav"],
    "KnowledgeCo":["rthk2theme.wav","knowledgecointro.wav"],
}

pairs={
    "happydaily":"https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/happydaily/m4a/{date}.m4a/index_0_a.m3u8",
    "healthpedia":"https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/healthpedia/m4a/{date}.m4a/index_0_a.m3u8",
    "morningsuite":"https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio2/morningsuite/m4a/{date}.m4a/index_0_a.m3u8",
    "KnowledgeCo":"https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio2/KnowledgeCo/m4a/{date}.m4a/index_0_a.m3u8",
}

schedule={
    "happydaily":{"begin": 10,"end":12},
    "healthpedia":{"begin": 13,"end":15},
    "morningsuite":{"begin": 6,"end":10},
    "KnowledgeCo":{"begin": 6,"end":10},
}

def url_ok(url):
 
 
    r = requests.get(url, stream=True)

    if r.ok:
        #content = next(r.iter_content(10))
        return True
    else:
        print(f"HTTP Error {r.status_code} - {r.reason}")
        return False

def download(url,target_file):
    if(os.path.exists(target_file)):
        print(f"file {target_file} already exists,skipping")
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

def timestamp_sanity_check(result,skip_reasonable_time_sequence_check):
    print(result)
    if(len(result) == 0):
        raise ValueError("result cannot be empty")
    
    for r in result:
        if(len(r) != 2):
            raise ValueError(f"each element in result must have 2 elements, got {r}")
        cur_start_time = r[0]
        cur_end_time = r[1]
        if(cur_start_time > cur_end_time):
            raise ValueError(f"start time {cur_start_time} is greater than end time {cur_end_time}")
        # program should last at least 15 minutes between half an hour interval news reports
        # TODO: still need to account for 15 hour interval news report at night time
        if not skip_reasonable_time_sequence_check:
            if(cur_end_time - cur_start_time < 15*60):
                raise TimeSequenceError(f"duration for program segment {cur_end_time - cur_start_time} seconds is less than 15 minutes")
    
    for i in range(1,len(result)):
        cur = result[i]
        cur_start_time = cur[0]
        prev = result[i-1]
        prev_end_time = prev[1]
        gap = cur_start_time - prev_end_time
        if(gap < 0):
            raise ValueError(f"start time {cur_start_time} is less than previous end time {prev_end_time}")
        # news report and commercial time should not be 10 minutes or longer
        elif(not skip_reasonable_time_sequence_check and gap >= 10*60):
            raise TimeSequenceError(f"gap between {cur_start_time} and {prev_end_time} is 10 minutes or longer")
        
    return result

# total_time is needed to set end time
# if it ends with intro instead of news report
# did a count down and the beep intro for news report is about 6 seconds
# skip_reasonable_time_sequence_check: skip sanity checks related to unreasonable duration or gaps, mainly for testing
# otherwise will have to rewrite lots of tests if the parameters changed
def process_timestamps(news_report,intro,total_time,news_report_second_pad=6,skip_reasonable_time_sequence_check=False):
    pair = []

    if len(news_report) != len(set(news_report)):
       raise ValueError("news report has duplicates, clean up duplicates first")   

    if len(intro) != len(set(intro)):
       raise ValueError("intro has duplicates, clean up duplicates first")   


    # will bug out if not sorted
    #news_report = deque([40,90,300])
    #intro =       deque([60,200,400])
    news_report=deque(sorted(news_report))
    intro=deque(sorted(intro))

    for i in intro:
        if i > total_time:
            raise ValueError(f"intro overflow, is greater than total time {total_time}")

    cur_intro = 0

    #news_report = deque([598, 2398, 3958, 5758])
    #intro = deque([1056, 2661, 4463])
    # no news report
    if(len(news_report)==0):
        # no need to trim
        return [[cur_intro, total_time]]
    
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

    pair=[]

    news_report_followed_by_intro = True
    while(len(news_report)>0):
        if(not news_report_followed_by_intro):
           raise ValueError("cannot have news report followed by news report")
        news_report_followed_by_intro=False
        cur_news_report = news_report.popleft()
        if(cur_intro > total_time):
            raise ValueError("intro overflow, is greater than total time {total_time}")
        pair.append([cur_intro, cur_news_report])
        # get first intro after news report
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
            #print("cur_news_report",cur_news_report,"total_time",total_time)
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
    
    timestamp_sanity_check(result,skip_reasonable_time_sequence_check=skip_reasonable_time_sequence_check)

    return result

def split_audio(input_file, output_file, start_time, end_time,total_time):
    #print( (str(datetime.timedelta(seconds=start_time)), str(datetime.timedelta(seconds=end_time))) )
    #return
    (
    ffmpeg.input(input_file, ss=str(datetime.timedelta(seconds=start_time)), to=str(datetime.timedelta(seconds=end_time)))
            .output(output_file,acodec='copy',vcodec='copy').overwrite_output().run()
    )

def concatenate_audio(input_files, output_file,tmpdir):
    list_file = os.path.join(tmpdir, 'list.txt')
    with open(list_file,'w') as f:
        for file_name in input_files:
            print(f"file {file_name}",file=f)

    artist="rthk"
    #title="tit'le"
    #album='al"bum'

    basename,extension = os.path.splitext(os.path.basename(output_file))

    album,date_str = extract_prefix(basename)
    title=basename

    # add artist, album and title metadata, can't think of better way than json dumps to escape
    # no need to oversolve it for now for quotes and equal signs
    metadata_list = ["title={}".format(title), "artist={}".format(artist), "album={}".format(album), ]
    metadata_dict = {f"metadata:g:{i}": e for i, e in enumerate(metadata_list)}

    (
        ffmpeg
            .input(list_file, format='concat', safe=0)
            .output(output_file, c='copy', **metadata_dict).overwrite_output().run()
    )


def md5file(file):
    with open(file, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def get_sec_from_str(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def extract_prefix(text):
  match = re.match(r"(.*\d{8,})", text)
  return (match.group(1)[:-8],match.group(1)[-8:]) if match else (None,None)

def scrape(input_file):
    
    # print(extension)
    # print(dir)
    # print(basename)
    #exit(1)
    basename,extension = os.path.splitext(os.path.basename(input_file))
    dir = os.path.dirname(input_file)
    print(basename)
    #md5=md5file(input_file)  # to get a printable str instead of bytes

    tsformatted = None

    jsonfile = f'{input_file}.json'
    if os.path.exists(jsonfile):
        tsformatted=json.load(open(jsonfile))

    total_time = math.ceil(float(ffmpeg.probe(input_file)["format"]["duration"]))
    print("total_time",total_time,"---")
    #exit(1)
    if not tsformatted:
        # Find clip occurrences in the full audio
        news_report_peak_times = find_clip_in_audio_in_chunks('./audio_clips/rthk_beep.wav', input_file, method="correlation")
        print(news_report_peak_times)

        #if any(basename.startswith(prefix) for prefix in ["happydaily","healthpedia"]):
        if any(basename.startswith(prefix) for prefix in ["happydaily"]):
            clips = introclips["happydaily"]
        elif any(basename.startswith(prefix) for prefix in ["healthpedia"]):
            clips = introclips["healthpedia"]
        elif any(basename.startswith(prefix) for prefix in ["morningsuite"]):
            clips = introclips["morningsuite"]
        elif any(basename.startswith(prefix) for prefix in ["KnowledgeCo"]):
            clips = introclips["KnowledgeCo"]
        else:
            raise NotImplementedError(f"not supported {basename}")
        program_intro_peak_times=[]
        for c in clips:
            print(f"Finding {c}")
            intros=find_clip_in_audio_in_chunks(f'./audio_clips/{c}', input_file, method="correlation",cleanup=False)
            print("intros",[str(datetime.timedelta(seconds=t)) for t in intros],"---")
            program_intro_peak_times.extend(intros)
        #program_intro_peak_times = cleanup_peak_times(program_intro_peak_times)
        # deduplicate
        program_intro_peak_times = list(sorted(dict.fromkeys([peak for peak in program_intro_peak_times])))
        print(program_intro_peak_times)
        print("program_intro_peak_times",[str(datetime.timedelta(seconds=t)) for t in program_intro_peak_times],"---")

        for offset in news_report_peak_times:
            print(f"Clip news_report_peak_times at the following times (in seconds): {str(datetime.timedelta(seconds=offset))}" )
        #    print(f"Offset: {offset}s" )
        
        for offset in program_intro_peak_times:
            print(f"Clip program_intro_peak_times at the following times (in seconds): {str(datetime.timedelta(seconds=offset))}" )
        #    #print(f"Offset: {offset}s" )
        pair = process_timestamps(news_report_peak_times, program_intro_peak_times,total_time)
        print("pair",pair)
        tsformatted = [[str(datetime.timedelta(seconds=t)) for t in sublist] for sublist in pair]
    else:
        pair = [[get_sec(t) for t in sublist] for sublist in tsformatted]
    print(pair)
    print(tsformatted)
    with open(jsonfile,'w') as f:
        f.write(json.dumps(tsformatted, indent=4))
    splits=[]

    output_dir= os.path.abspath(os.path.join(f"{dir}","trimmed"))
    output_file= os.path.join(output_dir,f"{basename}_trimmed{extension}")

    # if os.path.exists(output_file):
    #     print(f"file {output_file} already exists,skipping")
    #     return
    
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
    #path = os.path.join(tmp, 'something')
        for i,p in enumerate(pair):
            file_segment = os.path.join(tmpdir,f"{i+1}{extension}")
            start_time = p[0]
            end_time = p[1]
            split_audio(input_file, file_segment, start_time, end_time, total_time)
            splits.append(file_segment)
        concatenate_audio(splits, output_file,tmpdir)
        filename=os.path.basename(output_file)
        print(filename)
        dirname,date_str = extract_prefix(filename)
        print(dirname)
        dirname = '' if dirname is None else dirname
        path = f"/rthk/trimmed/{dirname}/{filename}"
        upload_file(output_file,path,skip_if_exists=True)

def is_time_after(current_time,hour):
  target_time = datetime.time(hour, 0, 0)  # Set minutes and seconds to 0
  return current_time > target_time

def download_and_scrape(download_only=False):
    
    #date = datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d")
    date = datetime.datetime.now(pytz.timezone('Asia/Hong_Kong'))
    date_str=date.strftime("%Y%m%d")
    for key, urltemplate in pairs.items():
        url = urltemplate.format(date=date_str)
        print(key)
        end_time = schedule[key]["end"]
        if not is_time_after(date.time(),end_time):
            print(f"skipping {key} because it is not yet from {end_time}")
            continue
        elif not url_ok(url):
            print(f"skipping {key} because url {url} is not ok")
            continue
        dest_file = os.path.abspath(f"./tmp/{key}{date_str}.m4a")
        try:
            download(url,dest_file)
            upload_file(dest_file,f"/rthk/{os.path.basename(dest_file)}",skip_if_exists=True)
            if(download_only):
                continue
            scrape(dest_file)
        except Exception as e:
            print(f"error happened when processing for {key}",e)
            print(traceback.format_exc())
            continue

def command():
    parser = argparse.ArgumentParser()
    parser.add_argument('action')     
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')
    parser.add_argument('--pattern-file', metavar='audio file', type=str, help='pattern file to convert sample')
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()
    if(args.action == 'scrape'):
        input_file = args.audio_file
        scrape(input_file)
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
