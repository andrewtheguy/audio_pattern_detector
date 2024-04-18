
import argparse
from collections import deque
import datetime
import hashlib
import json
import os
import shutil
import string
import tempfile

import ffmpeg
import pytz

from audio_offset_finder_v2 import cleanup_peak_times, convert_audio_to_clip_format, find_clip_in_audio_in_chunks

introclips={
    "happydaily":["rthk1clip.wav"],
    "morningsuite":["morningsuitethemefemalevoice.wav","morningsuiteintromale.wav"],
}

def download(url,target_file):
    if(os.path.exists(target_file)):
        print("file {target_file} already exists,skipping")
        return
    print(f'downloading {target_file}')
    with tempfile.TemporaryDirectory() as tmpdir:
        basename,extension = os.path.splitext(os.path.basename(target_file))
    
        tmp_file = os.path.join(tmpdir,f"download{extension}")
        (
        ffmpeg.input(url).output(tmp_file, **{'bsf:a': 'aac_adtstoasc'}, c='copy', loglevel="error")
              .run()
        )
        shutil.move(tmp_file,target_file)
    print('downloaded')    

#returns none if no need to trim
def process(news_report,intro):
    pair = []
    placehold_for_max = None
    # will bug out if one not followed by another, i.e. intro followed by intro or news followed
    # by news
    #news_report = deque([40,90,300])
    #intro =       deque([60,200,400])
    news_report=deque(news_report)
    intro=deque(intro)

    #news_report = deque([598, 2398, 3958, 5758])
    #intro = deque([1056, 2661, 4463])
    # no news report
    if(len(news_report)==0):
        return None
    
    if(len(intro)==0): # has news report but no intro
        raise NotImplementedError("not handling it yet")
        #return
    
    minimum = min(news_report[0], intro[0])


    # intro first, fake news report happening at the beginning
    if(intro[0] == minimum):
        if(news_report[0]>minimum):
            news_report.appendleft(0)
    
    
    if(len(news_report)>len(intro)): # news report at the end, pad intro
        intro.append(placehold_for_max)

    if len(news_report)!=len(intro):
        raise ValueError("not the same length")
    print(news_report)
    print(intro)
    min_len = min(len(news_report), len(intro))
    print(min_len)
    print('---')
    print(news_report)
    print(intro)
    print('---')
    
    cur = 0
    for i in range(min_len):
        news_report_second_pad=4
        news_report_ts = news_report[i]
        #if(cur > 0):

        if(news_report_ts == intro[i] or cur == news_report_ts): # happening the same time, skipping
            cur = intro[i]
            continue

        # pad two seconds to news report
        next_intro = intro[i+1] if len(intro) > i+1 else None
        if next_intro and news_report_ts + news_report_second_pad <= next_intro:
            news_report_ts = news_report_ts + news_report_second_pad
            
        pair.append([cur, news_report_ts]) 
        cur = intro[i]

    return pair

        

def split_audio(input_file, output_file, start_time, end_time):
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

    if not tsformatted:

        # Find clip occurrences in the full audio
        news_report_peak_times = find_clip_in_audio_in_chunks('./audio_clips/rthk_beep.wav', input_file, method="correlation")
        print(news_report_peak_times)

        if any(basename.startswith(prefix) for prefix in ["happydaily","healthpedia"]):
            clips = introclips["happydaily"]
        elif any(basename.startswith(prefix) for prefix in ["morningsuite"]):
            clips = introclips["morningsuite"]
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
        pair = process(news_report_peak_times, program_intro_peak_times)
        tsformatted = [[str(datetime.timedelta(seconds=t)) for t in sublist] for sublist in pair]
    else:
        pair = [[get_sec(t) for t in sublist] for sublist in tsformatted]
    print(pair)
    print(tsformatted)
    with open(jsonfile,'w') as f:
        f.write(json.dumps(tsformatted, indent=4))
    splits=[]
    
    with tempfile.TemporaryDirectory() as tmpdir:
    #path = os.path.join(tmp, 'something')
        for i,p in enumerate(pair):
            new_filename = os.path.join(tmpdir,f"{i+1}{extension}")
            print(new_filename)
            output_file = new_filename
            start_time = p[0]
            end_time = p[1]
            split_audio(input_file, output_file, start_time, end_time)
            splits.append(output_file)
        #fdsfsd
        concatenate_audio(splits, os.path.abspath(os.path.join(f"{dir}",f"{basename}_trimmed{extension}")),tmpdir)

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
        date = datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d")
        #date = datetime.datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%Y%m%d")
        download(f"https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/happydaily/m4a/{date}.m4a/index_0_a.m3u8",
                        os.path.abspath(f"./tmp/happydaily{date}.m4a"))
        download(f"https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/healthpedia/m4a/{date}.m4a/index_0_a.m3u8",
                        os.path.abspath(f"./tmp/healthpedia{date}.m4a"))
        download(f"https://rthkaod2022.akamaized.net/m4a/radio/archive/radio2/morningsuite/m4a/{date}.m4a/index_0_a.m3u8",
                        os.path.abspath(f"./tmp/morningsuite{date}.m4a"))
        
    else:
        raise NotImplementedError(f"action {args.action} not implemented")


    
if __name__ == '__main__':
    #pair=[]
    #process(pair)
    #print(pair)
    command()
