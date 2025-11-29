from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from audio_pattern_detector.audio_utils import load_wave_file, DEFAULT_TARGET_SAMPLE_RATE


class ReadableStream(Protocol):
    """Protocol for objects that can read bytes."""
    def read(self, size: int, /) -> bytes: ...


@dataclass(frozen=True)
class AudioClip:
    name: str
    audio: NDArray[np.float32]
    sample_rate: int

    # only takes filename
    @staticmethod
    def from_audio_file(clip_path, sample_rate=None):
        """Load an audio clip from a file.

        Args:
            clip_path: Path to the audio file.
            sample_rate: Target sample rate for the clip. If None, uses DEFAULT_TARGET_SAMPLE_RATE (8000).

        Returns:
            AudioClip instance with the loaded audio.
        """
        if sample_rate is None:
            sample_rate = DEFAULT_TARGET_SAMPLE_RATE
        # Load the audio clip
        clip = load_wave_file(clip_path, expected_sample_rate=sample_rate)
        # convert to float
        #clip = convert_audio_arr_to_float(clip)
        clip_name = Path(clip_path).stem
        return AudioClip(name=clip_name, audio=clip, sample_rate=sample_rate)

    def clip_length_seconds(self):
        return len(self.audio) / self.sample_rate


@dataclass(frozen=True)
class AudioStream:
    name: str
    audio_stream: ReadableStream  # raw byte stream of float32 mono PCM audio at sample_rate
    sample_rate: int
