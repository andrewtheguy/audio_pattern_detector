

import argparse
import glob
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from venv import logger

import ffmpeg

from audio_offset_finder_v2 import DEFAULT_METHOD, find_clip_in_audio_in_chunks

from andrew_utils import seconds_to_time
from file_upload.upload_utils2 import sftp_file_exists, upload_file
from process_timestamps import process_timestamps_single_intro
from scrape import concatenate_audio, get_sec, split_audio_by_time_sequences
from utils import extract_prefix
from upload_utils import sftp

streams={
    "1810加油站": {
        "introclips": ["am1430/1810_add_oil_intro.wav","am1430/1810_add_oil_end.wav"],
        #"endingclips": ["am1430/1810_add_oil_end.wav"],
        "ends_with_intro": True,
        "expected_num_segments": 4,
    },
    "天空下的彩虹": {
        "introclips": ["am1430/天空下的彩虹intro.wav"],
        "ends_with_intro": True,
        "expected_num_segments": 3,
    },
    "漫談法律": {
        "introclips": ["am1430/漫談法律intro.wav"],
        "endingclips": ["am1430/opinion_only.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 4,
    },
    "日落大道": { # for recorded one
        "introclips": ["am1430/日落大道smallinterlude.wav","am1430/日落大道interlude.wav"],
        #"endingclips": ["am1430/thankyouwatchingsunset.wav"],
        "endingclips": [],
        "ends_with_intro": False,
        "expected_num_segments": 5,
    },
}

correlation_threshold_intro = 0.3

def scrape_single_intro(input_file,stream_name):
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

        audio_name,_ = os.path.splitext(os.path.basename(input_file))

        program_intro_peak_times=[]
        program_intro_peak_times_debug=[]
        clip_paths=[f'./audio_clips/{c}' for c in clips]
        intros_all=find_clip_in_audio_in_chunks(clip_paths, input_file,method=DEFAULT_METHOD,correlation_threshold=correlation_threshold_intro)
        for c in clips:
            clip_path=f'./audio_clips/{c}'
            intros=intros_all[clip_path]
            #print("intros",[seconds_to_time(seconds=t,include_decimals=False) for t in intros],"---")
            program_intro_peak_times.extend(intros)
            intros_debug = sorted(intros)
            program_intro_peak_times_debug.append({c:[intros_debug,[seconds_to_time(seconds=t,include_decimals=True) for t in intros_debug]]})
        #program_intro_peak_times = cleanup_peak_times(program_intro_peak_times)
        #logger.debug(program_intro_peak_times)
        print("program_intro_peak_times",[seconds_to_time(seconds=t,include_decimals=True) for t in sorted(program_intro_peak_times)],"---")

        ends_with_intro = stream["ends_with_intro"]
  
        if not ends_with_intro:
            ending_clips = stream["endingclips"]
            # find earliest ending or fallback to total_time
            ending = total_time
            endings_array = []
            for c in ending_clips:
                #print(f"Finding {c}")
                endings=find_clip_in_audio_in_chunks(f'./audio_clips/{c}', input_file,method=DEFAULT_METHOD,correlation_threshold=correlation_threshold_intro)
                #print("intros",[seconds_to_time(seconds=t,include_decimals=False) for t in intros],"---")
                endings_array.extend(endings)
            if len(endings_array)>0:
                ending = max(endings_array)
            print("ending",seconds_to_time(seconds=ending,include_decimals=True),"---")
        else:
            ending = None # will be calculated later

        expected_num_segments = stream["expected_num_segments"]

        pair = process_timestamps_single_intro(program_intro_peak_times,ending,ends_with_intro=ends_with_intro,expected_num_segments=expected_num_segments)
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

    #splits=[]

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
        concatenate_audio(splits, output_file_trimmed,tmpdir,channel_name="am1430")
        upload_path_trimmed = f"/am1430/trimmed/{dirname}/{filename_trimmed}"
        upload_file(output_file_trimmed,upload_path_trimmed,skip_if_exists=True)
        
    return output_dir_trimmed,output_file_trimmed

def command():
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')

    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()

    input_file = args.audio_file
    input_dir = os.path.dirname(input_file)
    #stream_name,date_str = extract_prefix(os.path.basename(input_file))

    stream_name = os.path.basename(input_dir)
    #print(stream_name)
    scrape_single_intro(input_file,stream_name=stream_name)



    
if __name__ == '__main__':
    #print(url_ok("https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/happydaily/m4a/20240417.m4a/index_0_a.m3u8"))
    
    #upload_file("./tmp/out.pcm","/test5/5.pcm",skip_if_exists=True)
    
    #exit(1)
    #pair=[]
    #process(pair)
    #print(pair)
    command()
