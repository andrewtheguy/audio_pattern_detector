
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
from webdav4.client import Client

from audio_offset_finder_v2 import cleanup_peak_times, convert_audio_to_clip_format, find_clip_in_audio_in_chunks
from sftp_utils import create_remote_dir_recursively

introclips={
    "happydaily":["happydailyfemaleintro.wav"],
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

# total_time is needed to set end time
# if it ends with intro instead of news report
def process(news_report,intro,total_time):
    pair = []
    # will bug out if not sorted
    # will bug out if one not followed by another, i.e. intro followed by intro or news followed
    # by news
    #news_report = deque([40,90,300])
    #intro =       deque([60,200,400])
    news_report=deque(news_report)
    intro=deque(intro)

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
    
    pair=[]

    news_report_followed_by_intro = True
    while(len(news_report)>0):
        if(not news_report_followed_by_intro):
           raise ValueError("cannot have news report followed by news report")
        news_report_followed_by_intro=False
        cur_news_report = news_report.popleft()
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
            print("cur_news_report",cur_news_report,"total_time",total_time)
            raise NotImplementedError("not handling news report not followed by intro yet unless news report is 10 seconds from the end to prevent missing an intro")
    print("before padding",pair)
    # did count down and the beeps are about 6 seconds
    news_report_second_pad=6
    for i,arr in enumerate(pair):
        cur_intro = arr[0]
        cur_news_report = arr[1]
        if(i+1>=len(pair)):
            next_intro = None
        else:
            next_intro = pair[i+1][0]
        # pad news_report_second_pad seconds to news report
        if(next_intro is None or cur_news_report + news_report_second_pad <= next_intro):
            arr[1] = cur_news_report + news_report_second_pad
    print("after padding",pair)

    return pair

# #returns none if no need to trim
# def process(news_report,intro):
#     pair = []
#     placehold_for_max = None
#     # will bug out if one not followed by another, i.e. intro followed by intro or news followed
#     # by news
#     #news_report = deque([40,90,300])
#     #intro =       deque([60,200,400])
#     news_report=deque(news_report)
#     intro=deque(intro)

#     #news_report = deque([598, 2398, 3958, 5758])
#     #intro = deque([1056, 2661, 4463])
#     # no news report
#     if(len(news_report)==0):
#         raise NotImplementedError("not handling it yet")
#         #return None
    
#     if(len(intro)==0): # has news report but no intro
#         raise NotImplementedError("not handling it yet")
#         #return
    
#     minimum = min(news_report[0], intro[0])


#     # intro first, fake news report happening at the beginning
#     if(intro[0] == minimum):
#         if(news_report[0]>minimum):
#             news_report.appendleft(0)
    
    
#     if(len(news_report)>len(intro)): # news report at the end, pad intro
#         intro.append(placehold_for_max)

#     if len(news_report)!=len(intro):
#         raise ValueError("not the same length")
#     print(news_report)
#     print(intro)
#     min_len = min(len(news_report), len(intro))
#     print(min_len)
#     print('---')
#     print(news_report)
#     print(intro)
#     print('---')
    
#     cur = 0
#     for i in range(min_len):
#         news_report_second_pad=4
#         news_report_ts = news_report[i]
#         #if(cur > 0):

#         if(news_report_ts == intro[i] or cur == news_report_ts): # happening the same time, skipping
#             cur = intro[i]
#             continue

#         # pad two seconds to news report
#         next_intro = intro[i+1] if len(intro) > i+1 else None
#         if next_intro and news_report_ts + news_report_second_pad <= next_intro:
#             news_report_ts = news_report_ts + news_report_second_pad
            
#         pair.append([cur, news_report_ts]) 
#         cur = intro[i]

#     return pair

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

    (
        ffmpeg
            .input(list_file, format='concat', safe=0)
            .output(output_file, c='copy').overwrite_output().run()
    )

def upload_file(file,dest_path,skip_if_exists=False):
    # create ssh client 
    with paramiko.SSHClient() as ssh_client:
        # remote server credentials
        host = "10.22.33.20"
        username = "andrew"
        password = "qwertasdfg"
        port = '2022'
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=host,port=port,username=username,password=password, look_for_keys=False)

        # create an SFTP client object
        with ssh_client.open_sftp() as sftp:
            if skip_if_exists:
                try:
                    sftp.stat(dest_path)
                    print(f'file {dest_path} already exists,skipping')
                    return
                except IOError:
                    print(f"file {dest_path} doesn't exist, uploading")
                    #return
            print(f"uploading {file} to {dest_path}")
            create_remote_dir_recursively(sftp_client=sftp, remote_dir=os.path.dirname(dest_path))
            sftp.put(file,dest_path)

    #client.upload_file(file,dest_path,overwrite=True)

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

# for testprefix20220414 it will return testprefix, and for testagain220220414 will return testagain2, not testagain
def extract_prefix(text):
  """Extracts the string before a date in YYYYMMDD format.

  Args:
    text: The input string.

  Returns:
    The extracted prefix, or None if no date is found.
  """
  match = re.match(r"(.*?)(?=\d{8})", text)
  return match.group(1) if match else None

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
        program_intro_peak_times = cleanup_peak_times(program_intro_peak_times)
        print(program_intro_peak_times)
        #print("program_intro_peak_times",[str(datetime.timedelta(seconds=t)) for t in program_intro_peak_times],"---")

        for offset in news_report_peak_times:
            print(f"Clip news_report_peak_times at the following times (in seconds): {str(datetime.timedelta(seconds=offset))}" )
        #    print(f"Offset: {offset}s" )
        
        for offset in program_intro_peak_times:
            print(f"Clip program_intro_peak_times at the following times (in seconds): {str(datetime.timedelta(seconds=offset))}" )
        #    #print(f"Offset: {offset}s" )
        pair = process(news_report_peak_times, program_intro_peak_times,total_time)
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
        dirname = extract_prefix(filename)
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
    
    upload_file("./tmp/out.pcm","/test4/nada/outchafa4.pcm",skip_if_exists=True)
    
    #exit(1)
    #pair=[]
    #process(pair)
    #print(pair)
    #command()
