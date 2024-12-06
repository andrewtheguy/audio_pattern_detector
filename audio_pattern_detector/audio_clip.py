import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from audio_pattern_detector.audio_utils import load_wave_file, convert_audio_arr_to_float, TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class AudioClip:
    name: str
    audio: NDArray[np.float32]
    sample_rate: int

    # only takes filename
    @staticmethod
    def from_audio_file(clip_path):
        # Load the audio clip
        clip = load_wave_file(clip_path, sr=TARGET_SAMPLE_RATE)
        # convert to float
        #clip = convert_audio_arr_to_float(clip)
        clip_name = Path(clip_path).stem
        return AudioClip(name=clip_name, audio=clip, sample_rate=TARGET_SAMPLE_RATE)


@dataclass(frozen=True)
class StreamingAudioClip:
    name: str
    audio_stream: io.BufferedReader # this should be raw byte stream of 16 bit mono 8000HZ PCM audio
    sample_rate: int
