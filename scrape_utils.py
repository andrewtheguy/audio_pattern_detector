import logging
import os
import subprocess

import ffmpeg

logger = logging.getLogger(__name__)

from andrew_utils import seconds_to_time
from utils import extract_prefix


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

