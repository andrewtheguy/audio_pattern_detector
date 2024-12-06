import io
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class AudioClip:
    name: str
    audio: NDArray[np.float32]

    @staticmethod
    def from_audio_file(clip_path):
        # Load the audio clip
        clip = load_audio_file(clip_path, sr=self.target_sample_rate)
        # convert to float
        clip = convert_audio_arr_to_float(clip)


@dataclass(frozen=True)
class StreamingAudioClip:
    name: str
    audio_stream: io.BufferedReader # this should be raw byte stream of 16 bit mono 8000HZ PCM audio
