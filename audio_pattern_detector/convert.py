import os

import numpy as np
from pydub import AudioSegment

from audio_pattern_detector.audio_utils import (
    convert_audio_file,
    convert_audio_arr_to_float,
)


def convert_audio_to_clip_format(audio_path, output_path):
    """Convert audio file to clip format (8kHz, mono)"""
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


def cmd_convert(args):
    """Handler for convert subcommand"""
    input_file = args.audio_file
    convert_audio_to_clip_format(input_file, args.dest_file)
