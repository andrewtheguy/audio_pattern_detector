
import argparse
from collections import deque
import datetime

from audio_offset_finder_v2 import find_clip_in_audio_in_chunks

# still WIP
def process(pair):
    
    # will bug out if one not followed by another, i.e. intro followed by intro or news followed
    # by news
    #news_report = deque([40,90,300])
    #intro =       deque([60,200,400])

    news_report = deque([598, 2398, 3958, 5758])
    intro = deque([1056, 2661, 4463])
    # no news report
    if(len(news_report)==0):
        #raise NotImplementedError("not handling it yet")
        return
    
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
        intro.append(9999999)

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
    #    #print(f"Offset: {offset}s" )
    
    for offset in program_intro_peak_times:
        print(f"Clip program_intro_peak_times at the following times (in seconds): {str(datetime.timedelta(seconds=offset))}" )
    #    #print(f"Offset: {offset}s" )
    
if __name__ == '__main__':
    pair=[]
    process(pair)
    print(pair)
    #scrape()
