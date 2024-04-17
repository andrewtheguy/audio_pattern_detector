
import argparse
from collections import deque
import datetime
import os
import tempfile

import ffmpeg

from audio_offset_finder_v2 import find_clip_in_audio_in_chunks

# still WIP
def process(news_report,intro):
    pair = []
    placehold_for_max = 9999999
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
        pair.append([0, placehold_for_max]) 
        #raise NotImplementedError("not handling it yet")
        return pair
    
    if(len(intro)==0): # has news report but no intro
        raise NotImplementedError("not handling it yet")
        #return
    
    minimum = min(news_report[0], intro[0])

    cur = 0

    # intro first, fake news report happening the same time
    if(intro[0] == minimum):
        if(news_report[0]>minimum):
            news_report.appendleft(minimum)
    
    
    if(len(news_report)>len(intro)): # news report at the end, pad intro
        intro.append(placehold_for_max)

    if len(news_report)!=len(intro):
        raise ValueError("not the same length")
    print(news_report)
    print(intro)
    min_len = min(len(news_report), len(intro))
    print(min_len)
    for i in range(min_len):
        if(news_report[i] == intro[i]): # happening the same time, skipping
            continue
        pair.append([cur, news_report[i]]) 
        cur = intro[i]
        #pair.append([news_report[i], intro[i]]) 

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
            .output(output_file, c='copy').run()
    )

def scrape():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()
    #print(args.method)

    # Find clip occurrences in the full audio
    news_report_peak_times = find_clip_in_audio_in_chunks('./audio_clips/rthk_beep.wav', args.audio_file, method="correlation")
    print(news_report_peak_times)

    # Find clip occurrences in the full audio
    program_intro_peak_times = find_clip_in_audio_in_chunks('./audio_clips/rthk1clip.wav', args.audio_file, method="correlation")
    print(program_intro_peak_times)

    for offset in news_report_peak_times:
        print(f"Clip news_report_peak_times at the following times (in seconds): {str(datetime.timedelta(seconds=offset))}" )
    #    print(f"Offset: {offset}s" )
    
    for offset in program_intro_peak_times:
        print(f"Clip program_intro_peak_times at the following times (in seconds): {str(datetime.timedelta(seconds=offset))}" )
    #    #print(f"Offset: {offset}s" )
    pair = process(news_report_peak_times, program_intro_peak_times)
    print(pair)
    splits=[]
    input_file = args.audio_file
    base, extension = os.path.splitext(input_file)
    with tempfile.TemporaryDirectory() as tmpdir:
    #path = os.path.join(tmp, 'something')
        for i,p in enumerate(pair):
            new_filename = os.path.join(tmpdir,f"{base}_{i+1}{extension}")
            print(new_filename)
            output_file = new_filename
            start_time = p[0]
            end_time = p[1]
            split_audio(input_file, output_file, start_time, end_time)
            splits.append(output_file)
        #fdsfsd
        concatenate_audio(splits, f"{base}_trimmed{extension}",tmpdir)

    
if __name__ == '__main__':
    #pair=[]
    #process(pair)
    #print(pair)
    scrape()
