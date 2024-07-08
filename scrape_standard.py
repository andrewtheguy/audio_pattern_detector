

import argparse
import datetime
import fcntl
import glob
import json
import math
import os
import traceback
from collections import defaultdict
from pathlib import Path
import shutil
import tempfile
from venv import logger

import ffmpeg
import pytz

from audio_offset_finder_v2 import DEFAULT_METHOD, AudioOffsetFinder

from andrew_utils import seconds_to_time, time_to_seconds
from file_upload.upload_utils2 import upload_file, remote_exists, download_file
from process_timestamps import process_timestamps_simple
from publish import publish_folder
from scrape_utils import concatenate_audio, split_audio_by_time_sequences
from utils import get_ffprobe_info

# clips should be non-repeating

# for those I grabbed myself
ripped_streams={
    "1810加油站": {
        "introclips": ["am1430/1810_add_oil_intro.wav","am1430/1810_add_oil_end.wav"],
        #"endingclips": ["am1430/1810_add_oil_end.wav"],
        "ends_with_intro": True,
        "min_duration": 60 * 60,  # guard against short recordings which resulted from failure
        "expected_num_segments": [3,4],
        "publish": True,
        "time":"2000",
        "wday": 0,  # 0-6, 0 is Monday
    },
    "天空下的彩虹": {
        "introclips": ["am1430/天空下的彩虹intro.wav"],
        "ends_with_intro": True,
        "min_duration": 60 * 60,  # guard against short recordings which resulted from failure
        "publish": True,
        "time":"1900",
        "wday": 2, # 0-6, 0 is Monday
    },
    "漫談法律": {
        "introclips": ["am1430/漫談法律intro.wav"],
        "endingclips": ["am1430/opinion_only2.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 4,
        "time":"1000",
        "publish": True,
        "wday": 6, # 0-6, 0 is Monday
    },
    "法律天地": {
        "introclips": ["am1430/法律天地intro.wav"],
        "endingclips": ["am1430/opinion_only2.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 1,
        "time": "1230",
        "wday": 5, # 0-6, 0 is Monday
        "publish": True,
    },
    "置業興家": {
        "introclips": ["am1430/置業興家intro2.wav"],
        "endingclips": ["am1430/opinion_only.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 1,
    },
    "日落大道": {
        "introclips": ["am1430/日落大道smallinterlude.wav",
                       "am1430/日落大道interlude.wav",
                       "am1430/日落大道intro1.wav",
                       "am1430/日落大道intro2.wav",
                       "am1430/日落大道interlude3.wav",
                       "am1430/日落大道interlude4.wav",
                       ],
        "endingclips": ["am1430/temple_bell.wav",
                        "am1430/programsponsoredby2.wav",
                        "am1430/programsponsoredby3.wav",
                        "am1430/trafficendsponsor.wav",
                        "am1430/wholecityjump.wav",
                        "am1430/thankyouwatchingsunset.wav"],
        #"endingclips": [],
        "ends_with_intro": False,
        "min_duration": 60 * 60 * 2,  # guard against short recordings which resulted from failure
        "expected_num_segments": [4,7],
        "time":"1600",
        "publish": True,
    },
    "受之有道": {
        "introclips": ["am1430/受之有道intro.wav"],
        "endingclips": ["am1430/受之有道outro.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 3,
        "min_duration": 60 * 60,  # guard against short recordings which resulted from failure
        "time":"1800",
        "publish": True,
    },
    "繼續有心人friday": {
        "introclips": ["am1430/繼續有心人intro.wav"],
        "endingclips": ["am1430/thankyouwatching繼續有心人.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 3,
        "min_duration": 60 * 60,  # guard against short recordings which resulted from failure
        # "expected_num_segments": 5,
        "time": "1200",
        "publish": True,
        "wday": 4, # 0-6, 0 is Monday
    },
}

recorded_streams={
    "日落大道": { # for recorded one
        "introclips": ["am1430/日落大道smallinterlude.wav","am1430/日落大道interlude.wav"],
        #"endingclips": ["am1430/thankyouwatchingsunset.wav"],
        "endingclips": [],
        "ends_with_intro": False,
        #"expected_num_segments": 5,
    },
}

#correlation_threshold_intro = 0.4

def scrape_single_intro(input_file,stream_name,recorded):
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
        target_streams = ripped_streams if not recorded else recorded_streams
        stream = target_streams[stream_name]
        min_duration = stream.get("min_duration",None)
        if min_duration and total_time<min_duration:
            raise ValueError(f"total_time {total_time} is less than min_duration {min_duration}")
        intro_clips = stream["introclips"]

        ends_with_intro = stream["ends_with_intro"]

        ending_clips=stream.get("endingclips",[])


        clip_paths=[f'./audio_clips/{c}' for c in intro_clips+ending_clips]

        program_intro_peak_times=[]

        peaks_all = AudioOffsetFinder(method=DEFAULT_METHOD, debug_mode=False,
                                       clip_paths=clip_paths).find_clip_in_audio(
            full_audio_path=input_file)

        for c in intro_clips:
            clip_path=f'./audio_clips/{c}'
            intros=peaks_all[clip_path]
            program_intro_peak_times.extend(intros)
        print("program_intro_peak_times",[seconds_to_time(seconds=t,include_decimals=True) for t in sorted(program_intro_peak_times)],"---")

        endings_array = []
        for c in ending_clips:
            clip_path=f'./audio_clips/{c}'
            endings=peaks_all[clip_path]
            endings_array.extend(endings)

        print("ending_peak_times",[seconds_to_time(seconds=t,include_decimals=True) for t in sorted(endings_array)],"---")    
    
        expected_num_segments = stream.get("expected_num_segments")

        pair = process_timestamps_simple(program_intro_peak_times,endings_array,ends_with_intro=ends_with_intro,total_time=total_time,
                                         expected_num_segments=expected_num_segments,
                                         intro_max_repeat_seconds=60, # consolidate close by intros
                                         )
        #print("pair before rehydration",pair)
        tsformatted = [[seconds_to_time(seconds=t,include_decimals=True) for t in sublist] for sublist in pair]
        #duration = [seconds_to_time(t[1]-t[0]) for t in pair]
        #gaps=[]
        #for i in range(1,len(pair)):
        #    gaps.append(seconds_to_time(pair[i][0]-pair[i-1][1]))
        peaks_save = [(clip_name, [seconds_to_time(seconds=t,include_decimals=True) for t in peaks]) for clip_name,peaks in peaks_all.items()]
        with open(jsonfile,'w') as f:
            f.write(json.dumps({"tsformatted": tsformatted,"peaks_all":peaks_save}, indent=4, ensure_ascii=False))
    else:
        pair = [[time_to_seconds(t) for t in sublist] for sublist in tsformatted]
        #print("pair after rehydration",pair)
    print("tsformatted",tsformatted)


    #splits=[]

    output_dir_trimmed= os.path.abspath(os.path.join(f"./tmp","trimmed",stream_name))
    output_file_trimmed= os.path.join(output_dir_trimmed,f"{basename}_trimmed{extension}")

    # if os.path.exists(output_file):
    #     print(f"file {output_file} already exists,skipping")
    #     return
    
    os.makedirs(output_dir_trimmed, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename_trimmed=os.path.basename(output_file_trimmed)
        dirname=stream_name
        splits=split_audio_by_time_sequences(input_file,total_time,pair,tmpdir)
        concatenate_audio(splits, output_file_trimmed,tmpdir,channel_name="am1430",total_time=total_time)
        upload_path_trimmed = f"/am1430/trimmed/{dirname}/{filename_trimmed}"
        upload_file(output_file_trimmed,upload_path_trimmed,skip_if_exists=True)
        
    return output_file_trimmed,jsonfile


num_podcast_to_keep = 3

def download_am1430():
    max_go_back_days = 7
    try:
        lockfile = open(f'./tmp/lockfile', "a+")
        fcntl.lockf(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        raise RuntimeError('can only run one instance at a time')

    #original_base_dir = os.path.abspath(f"./tmp/original")
    #shutil.rmtree(original_base_dir, ignore_errors=True)

    local_path_downloaded = defaultdict(list)

    num_to_download = num_podcast_to_keep
    if num_to_download < 1:
        raise ValueError("num_to_download must be greater than or equal to 1")
    for key, stream in ripped_streams.items():
        if ("publish" not in stream) or (not stream["publish"]):
            print(f"skipping {key} because publish is not set to True")
            continue
        else:
            pass
            #print(f"downloading {key}")

        time = stream["time"]
        num_downloaded = 0
        original_dir = os.path.abspath(f"./tmp/original/{key}")
        os.makedirs(original_dir, exist_ok=True)

        for glob_pattern in ["*.m4a","*.m4a.json"]:
            files_del = sorted(glob.glob(os.path.join(original_dir, glob_pattern)),reverse=True)[num_to_download:]
            for file_del in files_del:
                print(f"deleting {file_del}")
                Path(file_del).unlink(missing_ok=True)

        for days_ago in range(max_go_back_days):
            if num_downloaded >= num_to_download:
                break
            #raise "cahjfa"
            date = datetime.datetime.now(pytz.timezone('America/Los_Angeles')) - datetime.timedelta(days=days_ago)
            date_str = date.strftime("%Y%m%d")
            file_name = f"{key}{date_str}_{time}_s_1.m4a"
            dest_file = os.path.join(original_dir, file_name)
            only_this_wday = stream.get("wday",None)
            if only_this_wday:
                if date.weekday() != only_this_wday:
                    print(f"skipping {key} because wday is not {only_this_wday}")
                    continue
                folder = "single"
            else:
                folder = "multiple"
            dest_remote_path = f"/grabradiostreamed/am1430/{folder}/{key}/{file_name}"
            if remote_exists(dest_remote_path):
                print(f'file {dest_remote_path} already exists,downloading from {date_str} instead')
                download_file(dest_remote_path, dest_file)
                #num_downloaded += 1
            elif not os.path.exists(dest_file):
                print(f'file {dest_remote_path} does not exist on remote or local, skipping')
                continue
            #clip_length_second_stream = float(get_ffprobe_info(url)['format']['duration'])
            remote_jsonfile = f'{dest_remote_path}.json'
            jsonfile = f'{dest_file}.json'
            if remote_exists(remote_jsonfile) and not os.path.exists(jsonfile):
                download_file(remote_jsonfile, jsonfile)
            local_path_downloaded[key].append(dest_file)
            num_downloaded += 1
    return local_path_downloaded


def process_podcasts():
    try:
        lockfile = open(f'./tmp/lockfile', "a+")
        fcntl.lockf(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        raise RuntimeError('can only run one instance at a time')

    failed_scrape_files = []

    error_occurred = False

    local_path_downloaded = download_am1430()
    podcasts_publish = []
    for stream_name, dest_files in local_path_downloaded.items():  # one stream at a time
        num_success = 0
        output_dir_trimmed = None
        for dest_file in dest_files:
            try:
                output_dir_trimmed = os.path.abspath(os.path.join(f"./tmp", "trimmed", stream_name))
                output_file_trimmed,jsonfile = scrape_single_intro(dest_file, stream_name=stream_name, recorded=False)
                if "/multiple/" in dest_file:
                    folder = "multiple"
                else:
                    folder = "single"
                upload_file(jsonfile, f"/grabradiostreamed/am1430/{folder}/{stream_name}/{os.path.basename(dest_file)}.json",
                            skip_if_exists=True)
                #podcasts_publish.append(output_dir_trimmed)
                num_success += 1
            except Exception as e:
                print(f"error happened when processing for {stream_name}", e)
                print(traceback.format_exc())
                error_occurred = True
                failed_scrape_files.append({"file": dest_file, "error": str(e)})
                #continue
        if num_success <= 0:
            print(f"error happened when processing all files, skipping publishing podcasts for {stream_name}")
            continue
        else:
            podcasts_publish.append(output_dir_trimmed)
    num_to_publish = num_podcast_to_keep
    for podcast in podcasts_publish:
        print(f"publishing podcast {podcast} after scraping")
        # assuming one per day
        publish_folder(podcast, files_to_publish=num_to_publish, delete_old_files=True)

    if failed_scrape_files:
        print(f"failed to scrape the following files:")
        for hash in failed_scrape_files:
            print("file", hash["file"], "error", hash["error"], "---")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')
    parser.add_argument('--audio-folder', metavar='audio folder', type=str, help='audio folder to find pattern')

    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()

    if(args.action == 'scrape'):
        audio_folder = args.audio_folder
        if audio_folder:
            if args.audio_file:
                raise ValueError("please specify only one of audio-file or folder")
            for input_file in glob.glob(os.path.join(audio_folder, "*.m4a")):
                print("processing", input_file)
                input_dir = os.path.dirname(input_file)
                recorded = "recorded" in input_dir
                if recorded:
                    raise ValueError("recorded not supported for folder yet")
                stream_name = os.path.basename(input_dir)
                try:
                    scrape_single_intro(input_file, stream_name=stream_name, recorded=recorded)
                except Exception as e:
                    print(f"error happened when processing for {stream_name}", e)
                    print(traceback.format_exc())
        else:
            input_file = args.audio_file
            input_dir = os.path.dirname(input_file)
            # stream_name,date_str = extract_prefix(os.path.basename(input_file))
            recorded = "recorded" in input_dir

            stream_name = os.path.basename(input_dir)
            # print(stream_name)
            scrape_single_intro(input_file, stream_name=stream_name, recorded=recorded)
    elif(args.action == 'download'):
        download_am1430()
    elif(args.action == 'process_podcasts'):
        process_podcasts()
    else:
        raise ValueError(f"unknown action {args.action}")


