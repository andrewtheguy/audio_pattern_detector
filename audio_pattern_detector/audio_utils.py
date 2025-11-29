import math
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

# Default sample rate for audio pattern detection (8kHz).
# All audio clips and streams must use the same sample rate for matching to work.
DEFAULT_TARGET_SAMPLE_RATE = 8000

# Cache for ffmpeg availability check
_ffmpeg_available = None


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system.

    Returns:
        bool: True if ffmpeg is available, False otherwise.
    """
    global _ffmpeg_available
    if _ffmpeg_available is not None:
        return _ffmpeg_available

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        _ffmpeg_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        _ffmpeg_available = False

    return _ffmpeg_available


def load_wav_file_scipy(file_path: str) -> tuple[NDArray[np.float32], int]:
    """Load WAV file without ffmpeg using scipy.io.wavfile.

    Args:
        file_path: Path to WAV file.

    Returns:
        tuple: (audio_data as float32 normalized to [-1,1], sample_rate)

    Raises:
        ValueError: If file is not a valid WAV file or has unsupported format.
    """
    from scipy.io import wavfile

    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read WAV file {file_path}: {e}") from e

    # Convert to float32 normalized to [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        pass  # Already float32
    elif data.dtype == np.float64:
        data = data.astype(np.float32)
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV dtype: {data.dtype}")

    # Handle stereo -> mono if needed
    if len(data.shape) > 1:
        data = data.mean(axis=1).astype(np.float32)

    return data, sample_rate


def resample_audio(audio: NDArray[np.float32], orig_sr: int, target_sr: int) -> NDArray[np.float32]:
    """Resample audio using scipy (no ffmpeg needed).

    Args:
        audio: Audio data as numpy array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio data.
    """
    if orig_sr == target_sr:
        return audio

    from scipy import signal

    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples)
    return np.asarray(resampled, dtype=np.float32)


def slicing_with_zero_padding(array: NDArray[np.floating[Any]], width: int, middle_index: int) -> NDArray[np.floating[Any]]:
    padding = width / 2

    beg = int(middle_index - math.floor(padding))
    end = int(middle_index + math.ceil(padding))

    if beg < 0:
        end = end - beg
        array = np.pad(array, (-beg, 0), 'constant')
        beg = beg - beg

    if end > len(array):
        array = np.pad(array, (0, end - len(array)), 'constant')
    # slice
    return np.array(array[beg:end])


def convert_audio_file(file_path: str, sr: int | None = None) -> NDArray[np.float32]:
    """Convert audio file to float32 PCM using ffmpeg."""
    with ffmpeg_get_float32_pcm(file_path, target_sample_rate=sr, ac=1) as stdout:
        data = stdout.read()
    return np.frombuffer(data, dtype=np.float32)

def load_wave_file(file_path: str, expected_sample_rate: int) -> NDArray[np.float32]:
    """Load wave file into float32 array.

    For WAV files, uses scipy directly (no ffmpeg needed).
    For non-WAV files, uses ffmpeg for conversion.
    Resampling is done if sample rate doesn't match the expected rate.

    Args:
        file_path: Path to audio file.
        expected_sample_rate: Expected sample rate of the output.

    Returns:
        numpy array of float32 audio samples at expected_sample_rate.
    """
    # For WAV files, use scipy directly (no ffmpeg needed)
    if file_path.lower().endswith('.wav'):
        data, sample_rate = load_wav_file_scipy(file_path)

        # Resample if needed
        if sample_rate != expected_sample_rate:
            data = resample_audio(data, sample_rate, expected_sample_rate)

        return data

    # For non-WAV files, use ffmpeg
    if not is_ffmpeg_available():
        raise ValueError(
            f"ffmpeg not available and file {file_path} is not a WAV file. "
            "Install ffmpeg or use WAV files for patterns."
        )

    return _load_wave_file_ffmpeg_convert(file_path, expected_sample_rate)


def _load_wave_file_ffmpeg_convert(file_path: str, target_sample_rate: int) -> NDArray[np.float32]:
    """Load and convert wave file using ffmpeg to target sample rate."""
    with ffmpeg_get_float32_pcm(file_path, target_sample_rate=target_sample_rate, ac=1) as stdout:
        data = stdout.read()

    # ffmpeg f32le output is already normalized to [-1, 1]
    samples: NDArray[np.float32] = np.frombuffer(data, dtype=np.float32)
    return samples


def downsample_preserve_maxima(curve: NDArray[np.floating[Any]], num_samples: int) -> NDArray[np.float32]:
    """Downsample a curve while preserving local maxima."""
    n_points = len(curve)
    step_size = n_points / num_samples
    compressed_curve: list[np.floating[Any]] = []

    for i in range(num_samples):
        start_index = int(i * step_size)
        end_index = int((i + 1) * step_size)

        if start_index >= n_points:
            break

        window = curve[start_index:end_index]
        if len(window) == 0:
            continue

        local_max_index = np.argmax(window)
        compressed_curve.append(window[local_max_index])

    # Adjust the length if necessary by adding the last element of the original curve
    if len(compressed_curve) < num_samples and len(curve) > 0:
        compressed_curve.append(curve[-1])

    if len(compressed_curve) != num_samples:
        raise ValueError(f"downsampled curve length {len(compressed_curve)} not equal to num_samples {num_samples}")

    return np.array(compressed_curve, dtype=np.float32)

@contextmanager
def ffmpeg_get_float32_pcm(
    full_audio_path: str,
    target_sample_rate: int | None = None,
    ac: int | None = None,
    from_stdin: bool = False,
    input_format: str | None = None,
) -> Generator[IO[bytes], None, None]:
    """Convert audio to float32 PCM with streaming output."""
    # Construct the ffmpeg command
    command = ["ffmpeg"]

    # When reading from stdin, we may need to specify input format
    if from_stdin:
        if input_format:
            command.extend(["-f", input_format])
        command.extend(["-i", "pipe:0"])
    else:
        command.extend(["-i", full_audio_path])

    command.extend([
        "-f", "f32le",  # Output format: 32-bit float little-endian
        "-acodec", "pcm_f32le",  # Audio codec
    ])

    if ac is not None:
        command.extend(["-ac", str(ac)])

    if target_sample_rate is not None:
        command.extend(["-ar", str(target_sample_rate)])

    command.extend([
                "-loglevel", "error",  # Suppress extra logs
                "pipe:"  # Output to stdout
                ])

    process = None

    try:
        # Run the command, capturing only stdout
        # When reading from stdin, connect parent's stdin to ffmpeg
        process = subprocess.Popen(
            command,
            stdin=sys.stdin.buffer if from_stdin else None,
            stdout=subprocess.PIPE  # Pipe stdout
        )
        assert process.stdout is not None  # guaranteed by stdout=subprocess.PIPE
        yield process.stdout
        if process.wait() != 0:
            raise ValueError(f"ffmpeg command failed with return code {process.returncode}")
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()


def write_wav_file(filepath: str, audio_data: NDArray[np.float32], sample_rate: int) -> None:
    """Write audio data to a wav file using ffmpeg.

    Args:
        filepath: Output file path
        audio_data: numpy array of float32 audio data in range [-1, 1]
        sample_rate: Sample rate in Hz
    """
    # Use ffmpeg to write wav file, accepting float32 directly
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-f", "f32le",  # Input format: 32-bit float little-endian
        "-ar", str(sample_rate),  # Sample rate
        "-ac", "1",  # Mono
        "-i", "pipe:",  # Read from stdin
        "-loglevel", "error",
        filepath,
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,  # ffmpeg writes to file, not stdout
    )
    process.communicate(input=audio_data.tobytes())

    if process.returncode != 0:
        raise ValueError(f"ffmpeg write failed with return code {process.returncode}")

def get_audio_duration(audio_path: str) -> float | None:
    """Get the duration of an audio file/URL using ffprobe.

    Args:
        audio_path: Path to audio file or URL

    Returns:
        Duration in seconds, or None if duration cannot be determined (e.g., live stream)
    """
    import json

    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        audio_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed: {result.stderr}")

    probe_data = json.loads(result.stdout)
    duration_str = probe_data.get("format", {}).get("duration")

    if duration_str is None:
        return None

    return float(duration_str)


def seconds_to_time(seconds: float, include_decimals: bool = True) -> str:
    """Convert seconds to time string format HH:MM:SS.mmm or HH:MM:SS."""
    if include_decimals:
        milliseconds = round(seconds * 1000)
        minutes_remaining, remaining_milliseconds = divmod(milliseconds, 60000)
        hours, minutes = divmod(minutes_remaining, 60)
        remaining_seconds = f"{remaining_milliseconds:05d}"
        remaining_seconds = remaining_seconds[:-3] + '.' + remaining_seconds[-3:]
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds}"
    else:
        seconds = round(seconds)
        minutes_remaining, remaining_seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes_remaining, 60)
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"