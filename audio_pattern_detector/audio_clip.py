from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from audio_pattern_detector.audio_utils import (
    load_wave_file,
    load_wav_from_bytes,
    resample_audio,
    DEFAULT_TARGET_SAMPLE_RATE,
)
from audio_pattern_detector.pattern_config import APD_EXTENSION, load_apd_file


class ReadableStream(Protocol):
    """Protocol for objects that can read bytes."""
    def read(self, size: int, /) -> bytes: ...


@dataclass(frozen=True)
class AudioClip:
    name: str
    audio: NDArray[np.float32]
    sample_rate: int
    # Non-None when the clip was loaded from an `.apd.toml` pattern config. Drives
    # strategy-based dispatch in AudioPatternDetector (e.g. "marker_tone").
    strategy: str | None = None
    strategy_params: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_audio_file(clip_path: str | Path, sample_rate: int | None = None) -> "AudioClip":
        """Load an audio clip from a file.

        Dispatches on extension: `.apd.toml` files are parsed as pattern
        configs (TOML) and the clip is synthesised from the declared generator;
        all other extensions go through the standard audio decoder.

        Args:
            clip_path: Path to the audio or .apd.toml file.
            sample_rate: Target sample rate for the clip. If None, uses DEFAULT_TARGET_SAMPLE_RATE (8000).

        Returns:
            AudioClip instance with the loaded audio.
        """
        if sample_rate is None:
            sample_rate = DEFAULT_TARGET_SAMPLE_RATE

        path_str = str(clip_path)
        if path_str.lower().endswith(APD_EXTENSION):
            # Strip the full compound extension (e.g. "rthk_beep.apd.toml" -> "rthk_beep")
            clip_name = Path(path_str[: -len(APD_EXTENSION)]).name
            config = load_apd_file(clip_path, sample_rate=sample_rate)
            return AudioClip(
                name=clip_name,
                audio=config.audio,
                sample_rate=sample_rate,
                strategy=config.strategy,
                strategy_params=config.strategy_params,
            )

        clip_name = Path(clip_path).stem
        clip = load_wave_file(path_str, expected_sample_rate=sample_rate)
        return AudioClip(name=clip_name, audio=clip, sample_rate=sample_rate)

    @staticmethod
    def from_wav_bytes(
        wav_bytes: bytes,
        name: str,
        sample_rate: int | None = None,
    ) -> "AudioClip":
        """Load an audio clip from WAV bytes.

        Args:
            wav_bytes: WAV file content as bytes.
            name: Name for the audio clip.
            sample_rate: Target sample rate for the clip. If None, uses DEFAULT_TARGET_SAMPLE_RATE (8000).

        Returns:
            AudioClip instance with the loaded audio.
        """
        if sample_rate is None:
            sample_rate = DEFAULT_TARGET_SAMPLE_RATE

        audio, source_sr = load_wav_from_bytes(wav_bytes, name)

        # Resample if needed
        if source_sr != sample_rate:
            audio = resample_audio(audio, source_sr, sample_rate)

        return AudioClip(name=name, audio=audio, sample_rate=sample_rate)

    def clip_length_seconds(self) -> float:
        return len(self.audio) / self.sample_rate


@dataclass(frozen=True)
class AudioStream:
    name: str
    audio_stream: ReadableStream  # raw byte stream of float32 mono PCM audio at sample_rate
    sample_rate: int
