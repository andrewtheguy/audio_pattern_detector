import argparse
import os
import numpy as np
from pydub import AudioSegment

from audio_pattern_detector.audio_utils import convert_audio_file, convert_audio_arr_to_float


# will be mono
def convert_audio_to_clip_format(audio_path, output_path):
    target_sample_rate = 8000

    if not os.path.exists(audio_path):
        raise ValueError(f"Audio {audio_path} does not exist")

    # Load the audio clip
    clip = convert_audio_file(audio_path, sr=target_sample_rate)

    # convert to float
    clip = convert_audio_arr_to_float(clip)

    # Convert float32 [-1, 1] back to int16 for pydub
    clip_int16 = (clip * (2**15)).astype(np.int16)

    # Create AudioSegment from numpy array
    audio = AudioSegment(
        clip_int16.tobytes(),
        frame_rate=target_sample_rate,
        sample_width=2,  # 16-bit = 2 bytes
        channels=1
    )

    # Export to wav file
    audio.export(output_path, format="wav")


if __name__ == '__main__':
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to convert')
    parser.add_argument('--dest-file', metavar='audio file', type=str, help='dest saved file')
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()

    # python convert.py --audio-file  /Volumes/andrewdata/audio_test/knowledge_co_e_word_intro.wav --dest-file audio_clips/knowledge_co_e_word_intro.wav
    input_file = args.audio_file
    convert_audio_to_clip_format(input_file,args.dest_file)
