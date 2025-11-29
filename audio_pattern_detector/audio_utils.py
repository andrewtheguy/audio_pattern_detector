import math
import subprocess
import sys
from contextlib import contextmanager

import numpy as np
from numpy._typing import DTypeLike

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


def load_wav_file_scipy(file_path: str) -> tuple[np.ndarray, int]:
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


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
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
    return resampled.astype(np.float32)


def slicing_with_zero_padding(array,width,middle_index):
    padding = width/2

    beg = int(middle_index-math.floor(padding))
    end = int(middle_index+math.ceil(padding))

    if beg < 0:
        end = end - beg
        #middle_index = middle_index - beg
        array = np.pad(array, (-beg, 0), 'constant')
        beg = beg - beg


    if end > len(array):
        array = np.pad(array, (0, end - len(array)), 'constant')
    # slice
    return np.array(array[beg:end])


def convert_audio_file(file_path, sr=None):
    # Create ffmpeg process - output is float32 directly
    with ffmpeg_get_float32_pcm(file_path, target_sample_rate=sr, ac=1) as stdout:
        data = stdout.read()
    return np.frombuffer(data, dtype="float32")
    #return librosa.load(file_path, sr=sr, mono=True)  # mono=True ensures a single channel audio

# load wave file into float32
def load_wave_file(file_path, expected_sample_rate):
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


def _load_wave_file_ffmpeg_convert(file_path, target_sample_rate):
    """Load and convert wave file using ffmpeg to target sample rate."""
    # Use ffmpeg to convert audio data to target sample rate
    with ffmpeg_get_float32_pcm(file_path, target_sample_rate=target_sample_rate, ac=1) as stdout:
        data = stdout.read()

    # ffmpeg f32le output is already normalized to [-1, 1]
    samples = np.frombuffer(data, dtype=np.float32)
    return samples


def _load_wave_file_ffmpeg(file_path, expected_sample_rate):
    """Load wave file using ffmpeg (original implementation)."""
    import json

    # Use ffprobe to get audio metadata
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels,sample_rate,bits_per_sample,codec_name",
        "-of", "json",
        file_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed: {result.stderr}")

    probe_data = json.loads(result.stdout)
    if not probe_data.get("streams"):
        raise ValueError(f"No audio streams found in {file_path}")

    stream = probe_data["streams"][0]
    channels = stream.get("channels", 0)
    sample_rate = int(stream.get("sample_rate", 0))
    bits_per_sample = stream.get("bits_per_sample", 0)

    # Check if it meets the conditions
    if channels != 1:
        raise ValueError(f"The file is not mono. Channels: {channels}")
    if sample_rate != expected_sample_rate:
        raise ValueError(f"The sample rate is not {expected_sample_rate} Hz. Sample rate: {sample_rate}")
    if bits_per_sample != 16:
        raise ValueError(f"The file is not 16-bit. Bits per sample: {bits_per_sample}")

    # Use ffmpeg to read audio data as float32 PCM directly
    with ffmpeg_get_float32_pcm(file_path, target_sample_rate=expected_sample_rate, ac=1) as stdout:
        data = stdout.read()

    # ffmpeg f32le output is already normalized to [-1, 1]
    samples = np.frombuffer(data, dtype=np.float32)

    return samples

# from librosa.util.buf_to_float
def buf_to_float(
    x: np.ndarray, *, n_bytes: int = 2, dtype: DTypeLike = np.float32
) -> np.ndarray:
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer
    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``
    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """
    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = f"<i{n_bytes:d}"

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def convert_audio_arr_to_float(audio):
    #raise "chafa"
    return buf_to_float(audio, n_bytes=2, dtype='float32')


def downsample_preserve_maxima(curve, num_samples):
    n_points = len(curve)
    step_size = n_points / num_samples
    compressed_curve = []

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

        # # Find peaks within this window
        # peaks, _ = find_peaks(window)
        # if len(peaks) == 0:
        #     # If no peaks, simply downsample by taking the first point in the window
        #     compressed_curve.append(window[0])
        # else:
        #     # Select the local maxima point
        #     local_max_index = peaks[np.argmax(window[peaks])]
        #     compressed_curve.append(window[local_max_index])

    # Adjust the length if necessary by adding the last element of the original curve
    if len(compressed_curve) < num_samples and len(curve) > 0:
        compressed_curve.append(curve[-1])

    if len(compressed_curve) != num_samples:
        raise ValueError(f"downsampled curve length {len(compressed_curve)} not equal to num_samples {num_samples}")

    return np.array(compressed_curve)

# convert audio to float32 pcm with streaming output
@contextmanager
def ffmpeg_get_float32_pcm(
    full_audio_path, target_sample_rate=None, ac=None, from_stdin=False, input_format=None
):
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
        yield process.stdout
        if process.wait() != 0:
            raise ValueError(f"ffmpeg command failed with return code {process.returncode}")
    finally:
        if process is not None:
            process.stdout.close()


def write_wav_file(filepath, audio_data, sample_rate):
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


def seconds_to_time(seconds, include_decimals=True):
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