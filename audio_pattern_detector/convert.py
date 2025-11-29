import argparse
import os

from audio_pattern_detector.audio_utils import (
    convert_audio_file,
    write_wav_file,
)


def convert_audio_to_clip_format(audio_path: str, output_path: str) -> None:
    """Convert audio file to clip format (8kHz, mono)"""
    target_sample_rate = 8000

    if not os.path.exists(audio_path):
        raise ValueError(f"Audio {audio_path} does not exist")

    # Load the audio clip (already float32 from ffmpeg)
    clip = convert_audio_file(audio_path, sr=target_sample_rate)

    # Write to wav file using ffmpeg
    write_wav_file(output_path, clip, target_sample_rate)


def cmd_convert(args: argparse.Namespace) -> None:
    """Handler for convert subcommand"""
    input_file = args.audio_file
    convert_audio_to_clip_format(input_file, args.dest_file)
