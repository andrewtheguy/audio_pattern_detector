

import argparse
import json
import math
import os
import shutil
import tempfile
from venv import logger

import ffmpeg

from audio_offset_finder_v2 import DEFAULT_METHOD, find_clip_in_audio_in_chunks

from andrew_utils import seconds_to_time
from process_timestamps import process_timestamps_single_intro
from scrape import get_sec, split_audio_by_time_sequences
from utils import extract_prefix

streams={
    "漫談法律": {
        "introclips": ["am1430/漫談法律intro.wav"],
    }
}

correlation_threshold_intro = 0.4

def scrape_single_intro(input_file,stream_name,date_str):
    save_segments = True
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

        pair = process_timestamps_single_intro(program_intro_peak_times,total_time)
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
        #concatenate_audio(splits, output_file_trimmed,tmpdir)
        #upload_path_trimmed = f"/rthk/trimmed/{dirname}/{filename_trimmed}"
        #upload_file(output_file_trimmed,upload_path_trimmed,skip_if_exists=True)
        
        # save segments
        for item in splits:
            dirname_segment = os.path.abspath(f"./tmp/segments/{dirname}/{date_str}")
            os.makedirs(dirname_segment, exist_ok=True)
            filename_segment=os.path.basename(item["file_path"])
            save_path=f"{dirname_segment}/{filename_segment}"
            shutil.move(item["file_path"],save_path)
            #upload_file(item["file_path"],upload_path,skip_if_exists=True)
    return output_dir_trimmed,output_file_trimmed


def command():
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')

    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()

    input_file = args.audio_file
    stream_name,date_str = extract_prefix(os.path.basename(input_file))
    #stream_name = "漫談法律"
    scrape_single_intro(input_file,stream_name=stream_name,date_str=date_str)



    
if __name__ == '__main__':
    #print(url_ok("https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/happydaily/m4a/20240417.m4a/index_0_a.m3u8"))
    
    #upload_file("./tmp/out.pcm","/test5/5.pcm",skip_if_exists=True)
    
    #exit(1)
    #pair=[]
    #process(pair)
    #print(pair)
    command()
