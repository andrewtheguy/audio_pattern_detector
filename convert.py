import argparse
import os
import soundfile as sf

from audio_pattern_detector.audio_utils import load_audio_file, convert_audio_arr_to_float


# will be mono
def convert_audio_to_clip_format(audio_path, output_path):
    target_sample_rate = 8000

    if not os.path.exists(audio_path):
        raise ValueError(f"Audio {audio_path} does not exist")

    # Load the audio clip
    clip = load_audio_file(audio_path, sr=target_sample_rate)

    # convert to float
    clip = convert_audio_arr_to_float(clip)

    sf.write(output_path, clip, target_sample_rate)


if __name__ == '__main__':
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-file', metavar='audio file', type=str, help='audio file to convert')
    parser.add_argument('--dest-file', metavar='audio file', type=str, help='dest saved file')
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()

    # python convert.py --pattern-file  /Volumes/andrewdata/audio_test/knowledge_co_e_word_intro.wav --dest-file audio_clips/knowledge_co_e_word_intro.wav
    input_file = args.pattern_file
    convert_audio_to_clip_format(input_file,args.dest_file)
