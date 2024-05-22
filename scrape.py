
import argparse

import logging
import os
import shutil
import subprocess
import tempfile

import ffmpeg
import requests

from audio_offset_finder_v2 import convert_audio_to_clip_format, DEFAULT_METHOD

logger = logging.getLogger(__name__)

from andrew_utils import seconds_to_time
from utils import extract_prefix, get_ffprobe_info


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
    clip_length_second_stream = float(get_ffprobe_info(url)['format']['duration'])
    with tempfile.TemporaryDirectory() as tmpdir:
        basename,extension = os.path.splitext(os.path.basename(target_file))
        tmp_file = os.path.join(tmpdir,f"download{extension}")
        (
        ffmpeg.input(url).output(tmp_file, **{'bsf:a': 'aac_adtstoasc'}, c='copy', loglevel="error")
              .run()
        )
        clip_length_second_file = float(get_ffprobe_info(tmp_file)['format']['duration'])
        second_tolerate=0.1
        if abs(clip_length_second_file - clip_length_second_stream) > second_tolerate:
            raise ValueError(f"downloaded file duration {clip_length_second_file} does not match stream duration {clip_length_second_stream} by {second_tolerate} seconds")
        shutil.move(tmp_file,target_file)
    print(f'downloaded to {target_file}')

def split_audio(input_file, output_file, start_time, end_time,total_time,artist,album,title):

    metadata_list = ["title={}".format(title), "artist={}".format(artist), "album={}".format(album), ]
    metadata_dict = {f"metadata:g:{i}": e for i, e in enumerate(metadata_list)}
    #raise "chafa"
    (
    ffmpeg.input(input_file, ss=seconds_to_time(seconds=start_time,include_decimals=True), to=seconds_to_time(seconds=end_time,include_decimals=True))
            .output(output_file,acodec='copy',vcodec='copy', loglevel="error", **metadata_dict).overwrite_output().run()
    )

def concatenate_audio(input_files, output_file,tmpdir,channel_name, total_time):
    list_file = os.path.join(tmpdir, 'list.txt')
    with open(list_file,'w') as f:                                            
        for item in input_files:
            file_name = item["file_path"]
            print(f"file {file_name}",file=f)

    artist=channel_name

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
    for i in range(len(input_files)):
        duration = input_files[i]["end_time"]-input_files[i]["start_time"]
        #print(f"start_time {start_time}")
        #print(f"duration {duration}")
        end_time=round(start_time+duration*1000)
        end_time = round(total_time*1000) if end_time > total_time*1000 else end_time
        #print(f"end_time {end_time}")
        path1=seconds_to_time(seconds=input_files[i]["start_time"],include_decimals=False).replace(':','_')
        path2=seconds_to_time(seconds=input_files[i]["end_time"],include_decimals=False).replace(':','_')
        title=f"{path1}-{path2}"
        text += f""";FFMETADATA1
[CHAPTER]
TIMEBASE=1/1000
START={start_time}
END={end_time}
title={title}\n"""
        start_time = end_time

    ffmetadatafile = os.path.join(tmpdir, 'ffmetadatafile.txt')
    #print(text)
    with open(ffmetadatafile, "w") as myfile:
        myfile.write(text)

    result = subprocess.run([
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
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed")


def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def split_audio_by_time_sequences(input_file,total_time,pair,output_dir):
    splits=[]
    dirname,date_str = extract_prefix(input_file)
    dirname = '' if dirname is None else dirname
    artist=dirname
    basename,extension = os.path.splitext(os.path.basename(input_file))
    for i,p in enumerate(pair):
        start_time = p[0]
        end_time = p[1]
        path1=seconds_to_time(seconds=start_time,include_decimals=False).replace(':','_')
        path2=seconds_to_time(seconds=end_time,include_decimals=False).replace(':','_')
        title=f"{path1}-{path2}"
        filename=f"{title}{extension}"
        file_segment = os.path.join(output_dir,filename)
        split_audio(input_file, file_segment, start_time, end_time, total_time,artist=artist,album=date_str,title=title)
        splits.append({"file_path": file_segment,
                        "start_time": start_time,
                        "end_time": end_time,})
    return splits

if __name__ == '__main__':
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('--pattern-file', metavar='audio file', type=str, help='audio file to convert')
    parser.add_argument('--dest-file', metavar='audio file', type=str, help='dest saved file')
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()
    if(args.action == 'convert'):
        # python scrape.py convert --pattern-file  /Volumes/andrewdata/audio_test/knowledge_co_e_word_intro.wav --dest-file audio_clips/knowledge_co_e_word_intro.wav
        input_file = args.pattern_file
        convert_audio_to_clip_format(input_file,args.dest_file)
    else:
        raise ValueError(f"unknown action {args.action}")
