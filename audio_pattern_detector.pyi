from collections.abc import Callable, Sequence
from os import PathLike
from typing import Protocol, TypedDict, final

__version__: str
APD_EXTENSION: str
DEFAULT_TARGET_SAMPLE_RATE: int
DEFAULT_SECONDS_PER_CHUNK: int

StrPath = str | PathLike[str]

# Called as on_detected(clip_name, timestamp_seconds) for each detection.
DetectedCallback = Callable[[str, float], object]

class ReadableStream(Protocol):
    def read(self, size: int, /) -> bytes: ...

class ClipConfig(TypedDict):
    duration_seconds: float
    sliding_window_seconds: int

class DetectorConfig(TypedDict):
    default_seconds_per_chunk: int
    min_chunk_size_seconds: int
    sample_rate: int
    clips: dict[str, ClipConfig]

@final
class MatchResult:
    @property
    def detections(self) -> dict[str, list[float]]: ...
    @property
    def duration_seconds(self) -> float: ...

@final
class Detector:
    def __new__(
        cls,
        pattern_files: Sequence[StrPath],
        *,
        seconds_per_chunk: int | None = 60,
        target_sample_rate: int = 8000,
        height_min: float | None = None,
        debug: bool = False,
        debug_dir: StrPath = "./tmp",
    ) -> Detector: ...
    @property
    def clip_names(self) -> list[str]: ...
    @property
    def seconds_per_chunk(self) -> int: ...
    @property
    def target_sample_rate(self) -> int: ...
    def config(self) -> DetectorConfig: ...
    def match_file(
        self, audio_file: StrPath, *, on_detected: DetectedCallback | None = None
    ) -> MatchResult: ...
    def match_wav_stream(
        self, stream: ReadableStream, *, on_detected: DetectedCallback | None = None
    ) -> MatchResult: ...

def clip_name(pattern_file: StrPath) -> str: ...
