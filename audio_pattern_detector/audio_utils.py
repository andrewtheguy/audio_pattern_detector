import math
import subprocess
from contextlib import contextmanager

import numpy as np
from numpy._typing import DTypeLike


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
    # Create ffmpeg process
    with ffmpeg_get_16bit_pcm(file_path, target_sample_rate=sr, ac=1) as stdout:
        data = stdout.read()
    return np.frombuffer(data, dtype="int16")
    #return librosa.load(file_path, sr=sr, mono=True)  # mono=True ensures a single channel audio

# load wave file with pydub into float32
def load_wave_file(file_path, expected_sample_rate):
    from pydub import AudioSegment

    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Check if it meets the conditions
    if audio.channels != 1:
        raise ValueError(f"The file is not mono. Channels: {audio.channels}")
    if audio.frame_rate != expected_sample_rate:
        raise ValueError(f"The sample rate is not {expected_sample_rate} Hz. Sample rate: {audio.frame_rate}")
    if audio.sample_width != 2:  # 2 bytes = 16-bit
        raise ValueError(f"The file is not 16-bit. Sample width: {audio.sample_width} bytes")

    # Convert to numpy array and normalize to float32 [-1, 1]
    samples = np.array(audio.get_array_of_samples(), dtype='float32')
    samples = samples / (2**15)  # Normalize 16-bit to [-1, 1]

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

# convert audio to 16 bit pcm with streaming output
@contextmanager
def ffmpeg_get_16bit_pcm(full_audio_path,target_sample_rate=None,ac=None):
    # Construct the ffmpeg command
    command = [
        "ffmpeg",
        "-i", full_audio_path,
        "-f", "s16le",  # Output format
        "-acodec", "pcm_s16le",  # Audio codec
    ]

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
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE  # Pipe stdout
        )
        yield process.stdout
        if process.wait() != 0:
            raise ValueError(f"ffmpeg command failed with return code {process.returncode}")
    finally:
        if process is not None:
            process.stdout.close()


TARGET_SAMPLE_RATE = 8000

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