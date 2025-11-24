import os

from audio_pattern_detector.audio_utils import (
    convert_audio_file,
    convert_audio_arr_to_float,
    write_wav_file,
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

    # Write to wav file using ffmpeg
    write_wav_file(output_path, clip, target_sample_rate)


def cmd_convert(args):
    """Handler for convert subcommand"""
    input_file = args.audio_file
    convert_audio_to_clip_format(input_file, args.dest_file)
