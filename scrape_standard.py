

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

from audio_offset_finder_v2 import DEFAULT_METHOD, AudioOffsetFinder

from andrew_utils import seconds_to_time, time_to_seconds
from file_upload.upload_utils2 import upload_file
from process_timestamps import process_timestamps_simple
from scrape_utils import concatenate_audio, split_audio_by_time_sequences

# clips should be non-repeating because I am using the non-repeat method, too much headache to deal with repeating clips

# for those I grabbed myself
ripped_streams={
    "1810加油站": {
        "introclips": ["am1430/1810_add_oil_intro.wav","am1430/1810_add_oil_end.wav"],
        #"endingclips": ["am1430/1810_add_oil_end.wav"],
        "ends_with_intro": True,
        "expected_num_segments": 4,
    },
    "天空下的彩虹": {
        "introclips": ["am1430/天空下的彩虹intro.wav"],
        "ends_with_intro": True,
        #"expected_num_segments": 3,
    },
    "漫談法律": {
        "introclips": ["am1430/漫談法律intro.wav"],
        "endingclips": ["am1430/opinion_only2.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 4,
    },
    "法律天地": {
        "introclips": ["am1430/法律天地intro.wav"],
        "endingclips": ["am1430/opinion_only2.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 1,
    },
    "置業興家": {
        "introclips": ["am1430/置業興家intro2.wav"],
        "endingclips": ["am1430/opinion_only.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 1,
    },
    "日落大道": {
        "introclips": ["am1430/日落大道intro1.wav","am1430/日落大道intro2.wav"],
        "endingclips": ["am1430/programsponsoredby.wav","am1430/thankyouwatchingsunset.wav"],
        #"endingclips": [],
        "ends_with_intro": False,
        "min_duration": 60 * 60 * 2,  # guard against short recordings which resulted from failure
        #"expected_num_segments": 5,
    },
    "受之有道": {
        "introclips": ["am1430/受之有道intro.wav"],
        "endingclips": ["am1430/受之有道outro.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 3,
        "min_duration": 60 * 60,  # guard against short recordings which resulted from failure
        # "expected_num_segments": 5,
    },
    "繼續有心人friday": {
        "introclips": ["am1430/繼續有心人intro.wav"],
        "endingclips": ["am1430/thankyouwatching繼續有心人.wav"],
        "ends_with_intro": False,
        "expected_num_segments": 3,
        "min_duration": 60 * 60,  # guard against short recordings which resulted from failure
        # "expected_num_segments": 5,
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

        peaks_all = AudioOffsetFinder(method="non_repeating_correlation", debug_mode=False,
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
                                         intro_max_repeat_seconds=60,
                                         )
        #print("pair before rehydration",pair)
        tsformatted = [[seconds_to_time(seconds=t,include_decimals=True) for t in sublist] for sublist in pair]

    else:
        pair = [[time_to_seconds(t) for t in sublist] for sublist in tsformatted]
        #print("pair after rehydration",pair)
    print("tsformatted",tsformatted)
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
        dirname=stream_name
        splits=split_audio_by_time_sequences(input_file,total_time,pair,tmpdir)
        concatenate_audio(splits, output_file_trimmed,tmpdir,channel_name="am1430",total_time=total_time)
        upload_path_trimmed = f"/am1430/trimmed/{dirname}/{filename_trimmed}"
        upload_file(output_file_trimmed,upload_path_trimmed,skip_if_exists=True)
        
    return output_dir_trimmed,output_file_trimmed

if __name__ == '__main__':
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')

    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()

    input_file = args.audio_file
    input_dir = os.path.dirname(input_file)
    #stream_name,date_str = extract_prefix(os.path.basename(input_file))
    recorded = "recorded" in input_dir

    stream_name = os.path.basename(input_dir)
    #print(stream_name)
    scrape_single_intro(input_file,stream_name=stream_name,recorded=recorded)