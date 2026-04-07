import json
import logging
import math
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from operator import itemgetter
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from audio_pattern_detector.audio_clip import AudioClip, AudioStream
from audio_pattern_detector.audio_utils import (
    DEFAULT_TARGET_SAMPLE_RATE,
    resample_preserve_maxima,
    seconds_to_time,
    slicing_with_zero_padding,
    write_wav_file,
)
from audio_pattern_detector.detection_utils import (
    analyze_pure_tone_candidate,
    extract_padded_segment,
    get_pure_tone_frequency,
)
from audio_pattern_detector.numpy_encoder import NumpyEncoder

logger = logging.getLogger(__name__)

# Default seconds per chunk for sliding window processing
DEFAULT_SECONDS_PER_CHUNK = 60

# Clips shorter than this go through the normal path with 0-100% window
SHORT_CLIP_DURATION_THRESHOLD = 0.5  # seconds

# Clip name that triggers the pure tone verification path
PURE_TONE_CLIP_NAME = "rthk_beep"


# Type definitions
class ClipData(TypedDict):
    """Internal clip data structure."""
    clip: NDArray[np.float32]
    clip_name: str
    sliding_window: int
    correlation_clip: NDArray[np.float32]
    correlation_clip_absolute_max: np.floating[Any]


class ClipCache(TypedDict):
    """Cache for clip-related computed data."""
    downsampled_correlation_clips: dict[str, NDArray[np.float32]]
    downsampled_pearson_windows: dict[str, list[NDArray[np.float32]]]


class ClipConfig(TypedDict):
    """Configuration for a single clip in get_config output."""
    duration_seconds: float
    sliding_window_seconds: int


class DetectorConfig(TypedDict):
    """Return type for get_config method."""
    default_seconds_per_chunk: int
    min_chunk_size_seconds: int
    sample_rate: int
    clips: dict[str, ClipConfig]


# Type alias for pattern detection callback
PatternDetectedCallback = Callable[[str, float], None]


def _mean_squared_error(y_true: NDArray[np.floating[Any]], y_pred: NDArray[np.floating[Any]]) -> np.floating[Any]:
    """Simple MSE implementation to avoid sklearn dependency."""
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)

def _write_audio_file(filepath: str, audio_data: NDArray[np.float32], sample_rate: int) -> None:
    """Helper function to write audio to a wav file."""
    write_wav_file(filepath, audio_data, sample_rate)


class AudioPatternDetector:

    def __init__(self, audio_clips: list[AudioClip], debug_mode: bool = False, seconds_per_chunk: int | None = DEFAULT_SECONDS_PER_CHUNK, target_sample_rate: int | None = None, debug_dir: str = './tmp', height_min: float | None = None) -> None:
        """Initialize the audio pattern detector.

        Args:
            audio_clips: List of AudioClip instances to detect.
            debug_mode: Enable debug mode for additional output.
            seconds_per_chunk: Seconds per chunk for sliding window processing.
            target_sample_rate: Target sample rate for all audio. If None, uses DEFAULT_TARGET_SAMPLE_RATE (8000).
            debug_dir: Base directory for debug output files.
            height_min: Override minimum correlation peak height (default: 0.25).
        """
        self.audio_clips = audio_clips
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        self.height_min = height_min
        self.normalize = True
        self.target_sample_rate = target_sample_rate if target_sample_rate is not None else DEFAULT_TARGET_SAMPLE_RATE
        self._similarity_debug: defaultdict[str, list[tuple[int, np.floating[Any]]]] = defaultdict(list)

        clips_already = set()
        max_clip_length = 0
        for audio_clip in self.audio_clips:
            if audio_clip.name in clips_already:
                raise ValueError(f"clip {audio_clip.name} needs to be unique")
            if audio_clip.sample_rate != self.target_sample_rate:
                raise ValueError(f"clip {audio_clip.name} needs to be {self.target_sample_rate} sample rate")
            clips_already.add(audio_clip.name)
            clip_length = len(audio_clip.audio)
            if clip_length > max_clip_length:
                max_clip_length = clip_length

        if seconds_per_chunk is None or seconds_per_chunk < 1:
            # 2 seconds padding
            seconds_per_chunk = math.ceil(max_clip_length / self.target_sample_rate) * 2
            logger.warning(f"seconds_per_chunk is not set or less than 1, setting it to longest clip * 2 seconds, which is {seconds_per_chunk} seconds")

        # Validate seconds_per_chunk against all clips' sliding windows
        # Track the largest min_chunk_size across all clips
        max_min_chunk_size = 0
        for audio_clip in self.audio_clips:
            clip_seconds = len(audio_clip.audio) / self.target_sample_rate
            sliding_window = math.ceil(clip_seconds)
            min_chunk_size = sliding_window * 2
            if min_chunk_size > max_min_chunk_size:
                max_min_chunk_size = min_chunk_size
            if seconds_per_chunk < min_chunk_size:
                raise ValueError(
                    f"seconds_per_chunk {seconds_per_chunk} is too small for clip '{audio_clip.name}' "
                    f"(duration: {clip_seconds:.2f}s, sliding_window: {sliding_window}s, "
                    f"minimum chunk size: {min_chunk_size}s)"
                )
        self._min_chunk_size = max_min_chunk_size

        self.seconds_per_chunk = seconds_per_chunk

        if seconds_per_chunk != 60:
            logger.warning(f"seconds_per_chunk {seconds_per_chunk} is not 60 seconds, turning off debug mode because it was made for 60 seconds only")
            self.debug_mode = False

        # Pre-compute clip data that doesn't depend on audio_stream
        self._clip_datas: dict[str, ClipData] = {}
        self._pure_tone_frequencies: dict[str, float] = {}
        self._clip_cache: ClipCache = {
            "downsampled_correlation_clips": {},
            "downsampled_pearson_windows": {},
        }

        for audio_clip in self.audio_clips:
            clip = audio_clip.audio
            clip_name = audio_clip.name
            clip_seconds = len(clip) / self.target_sample_rate

            # Compute sliding window
            sliding_window = math.ceil(clip_seconds)
            if sliding_window != clip_seconds:
                print(f"adjusted sliding_window from {clip_seconds} to {sliding_window} for {clip_name}", file=sys.stderr)

            # Normalize clip if enabled
            if self.normalize:
                from native_helper import integrated_loudness, loudness_normalize
                sr = self.target_sample_rate
                block = clip_seconds if clip_seconds < 0.5 else 0.4
                loudness = integrated_loudness(clip, sr, block_size=block)
                clip = loudness_normalize(clip, loudness, -16.0)

            # Compute correlation
            correlation_clip, absolute_max = self._get_clip_correlation(clip)

            # Debug output for correlation
            if self.debug_mode:
                import matplotlib.pyplot as plt
                print(f"clip_length {clip_name}", len(clip), file=sys.stderr)
                print(f"clip_length {clip_name} seconds", len(clip) / self.target_sample_rate, file=sys.stderr)
                print("correlation_clip_length", len(correlation_clip), file=sys.stderr)
                graph_dir = f"{self.debug_dir}/graph/clip_correlation"
                os.makedirs(graph_dir, exist_ok=True)

                plt.figure(figsize=(10, 4))
                plt.plot(correlation_clip)
                plt.title('Cross-correlation of the audio clip itself')
                plt.xlabel('Lag')
                plt.ylabel('Correlation coefficient')
                plt.savefig(f'{graph_dir}/{clip_name}.png')
                plt.close()

                # Save the original correlation clip graph (used for comparison in slice graphs)
                graph_dir_original = f"{self.debug_dir}/graph/cross_correlation_slice_original/{clip_name}"
                os.makedirs(graph_dir_original, exist_ok=True)
                plt.figure(figsize=(10, 4))
                plt.plot(correlation_clip, color='orange')
                plt.title('Cross-correlation of the audio clip itself (original pattern)')
                plt.xlabel('Lag')
                plt.ylabel('Correlation coefficient')
                plt.savefig(f'{graph_dir_original}/{clip_name}.png')
                plt.close()

            self._clip_datas[clip_name] = {
                "clip": clip,
                "clip_name": clip_name,
                "sliding_window": sliding_window,
                "correlation_clip": correlation_clip,
                "correlation_clip_absolute_max": absolute_max,
            }

            if clip_name == PURE_TONE_CLIP_NAME:
                freq = get_pure_tone_frequency(clip, self.target_sample_rate)
                if freq is not None:
                    self._pure_tone_frequencies[clip_name] = freq

        # Pre-compute chunk_size (4 bytes per sample for float32, mono)
        self._chunk_size = int(self.seconds_per_chunk * self.target_sample_rate) * 4

    def get_config(self) -> DetectorConfig:
        """Return the computed configuration values from __init__.

        Returns:
            DetectorConfig: Configuration including seconds_per_chunk, chunk_size_bytes,
                  min_chunk_size_seconds, and per-clip data (duration, sliding_window).
        """
        clips_config: dict[str, ClipConfig] = {}
        for clip_name, clip_data in self._clip_datas.items():
            clip_duration = len(clip_data["clip"]) / self.target_sample_rate
            clips_config[clip_name] = {
                "duration_seconds": round(clip_duration, 6),
                "sliding_window_seconds": clip_data["sliding_window"],
            }

        return {
            "default_seconds_per_chunk": DEFAULT_SECONDS_PER_CHUNK,
            "min_chunk_size_seconds": self._min_chunk_size,
            "sample_rate": self.target_sample_rate,
            "clips": clips_config,
        }

    def find_clip_in_audio(
        self,
        audio_stream: AudioStream,
        on_pattern_detected: PatternDetectedCallback | None = None,
        accumulate_results: bool = True,
    ) -> tuple[dict[str, list[float]] | None, float]:
        """Find clip occurrences in audio stream.

        Args:
            audio_stream: The audio stream to search in
            on_pattern_detected: Optional callback function called when a pattern is detected.
                                 Signature: on_pattern_detected(clip_name: str, timestamp: float)
            accumulate_results: If False, don't accumulate peak_times (saves memory for streaming).
                               When False, returns None for peak_times.

        Returns:
            Tuple of (peak_times dict or None if accumulate_results=False, total_time in seconds)
        """
        # clip_paths = self.clip_paths
        #
        # if not os.path.exists(full_audio_path):
        #     raise ValueError(f"Full audio {full_audio_path} does not exist")

        if audio_stream.sample_rate != self.target_sample_rate:
            raise ValueError(f"full_streaming_audio_clip {audio_stream.name} needs to be {self.target_sample_rate} sample rate")

        # Initialize parameters
        previous_chunk = None  # Buffer to maintain continuity between chunks

        # Only allocate if we need to accumulate results
        if accumulate_results:
            all_peak_times = {audio_clip.name: [] for audio_clip in self.audio_clips}
        else:
            all_peak_times = None


        full_audio_name = audio_stream.name
        stdout = audio_stream.audio_stream

        i = 0

        # Reset per-run debug state
        self._similarity_debug = defaultdict(list)

        total_time = 0.0

        # Process audio in chunks
        while True:
            in_bytes = stdout.read(self._chunk_size)
            if not in_bytes:
                break
            chunk = np.frombuffer(in_bytes, dtype="float32")

            total_time += len(chunk) / self.target_sample_rate

            # Collect all matches from all clips for this chunk
            chunk_matches = []  # List of (timestamp, clip_name) tuples

            for audio_clip in self.audio_clips:
                clip_data = self._clip_datas[audio_clip.name]

                peak_times = self._process_chunk(chunk=chunk,
                                                 previous_chunk=previous_chunk,
                                                 index=i,
                                                 clip_data=clip_data,
                                                 )

                # Collect matches for sorting
                if on_pattern_detected and peak_times:
                    for timestamp in peak_times:
                        chunk_matches.append((timestamp, audio_clip.name))

                if all_peak_times is not None:
                    all_peak_times[audio_clip.name].extend(peak_times)

            # Call callback in timestamp order (monotonic output)
            if on_pattern_detected and chunk_matches:
                chunk_matches.sort(key=lambda x: x[0])  # Sort by timestamp
                for timestamp, clip_name in chunk_matches:
                    on_pattern_detected(clip_name, timestamp)

            # Update previous_chunk to current chunk
            previous_chunk = chunk
            i = i + 1

        if self.debug_mode:
            import matplotlib.pyplot as plt

            suffix = full_audio_name

            for audio_clip in self.audio_clips:
                #clip_name, _ = os.path.splitext(os.path.basename(clip_path))
                clip_name = audio_clip.name

                # similarity debug
                graph_dir = f"{self.debug_dir}/graph/mean_squared_error_similarity/{clip_name}"
                os.makedirs(graph_dir, exist_ok=True)

                x_coords = []
                y_coords = []

                for index,similarity in self._similarity_debug[clip_name]:
                    x_coords.append(index)
                    y_coords.append(similarity)

                plt.figure(figsize=(10, 4))
                # Create scatter plot
                plt.scatter(x_coords, y_coords)


                ylimit = max(0.01, np.median(y_coords))
                # Set the y limits
                plt.ylim(0, ylimit)

                # Adding titles and labels
                plt.title('Scatter Plot for Similarity')
                plt.xlabel('Value')
                plt.ylabel('Sublist Index')
                plt.savefig(
                    f'{graph_dir}/{suffix}.png')
                plt.close()


        return all_peak_times, total_time

    def _get_clip_correlation(self, clip: NDArray[np.float32]) -> tuple[NDArray[np.float32], np.floating[Any]]:
        # Cross-correlate and normalize correlation
        from fft_correlation import fft_correlate_1d
        correlation_clip: NDArray[np.float32] = fft_correlate_1d(clip, clip, mode='full')

        # abs
        correlation_clip = np.abs(correlation_clip)
        absolute_max: np.floating[Any] = np.max(correlation_clip)
        correlation_clip /= absolute_max

        return correlation_clip, absolute_max


    # sliding_window: for previous_chunk in seconds from end
    # index: for debugging by saving a file for audio_section
    # seconds_per_chunk: default seconds_per_chunk
    def _process_chunk(
        self,
        chunk: NDArray[np.float32],
        clip_data: ClipData,
        previous_chunk: NDArray[np.float32] | None,
        index: int,
    ) -> list[float]:
        clip, _, sliding_window = itemgetter("clip","clip_name","sliding_window")(clip_data)
        sr = self.target_sample_rate
        seconds_per_chunk = self.seconds_per_chunk
        clip_seconds = len(clip) / sr
        chunk_seconds = len(chunk) / sr
        # Concatenate previous chunk for continuity in processing
        if previous_chunk is not None:
            if chunk_seconds < seconds_per_chunk:  # too small
                # no need for sliding window since it is the last piece
                subtract_seconds = -(chunk_seconds - seconds_per_chunk)
                audio_section_temp = np.concatenate((previous_chunk, chunk))[(-seconds_per_chunk * sr):]
                audio_section = audio_section_temp
            else:
                subtract_seconds = sliding_window
                audio_section = np.concatenate((previous_chunk[int(-sliding_window * sr):], chunk))
        else:
            subtract_seconds = 0
            audio_section = chunk

        if self.normalize:
            from native_helper import integrated_loudness, loudness_normalize
            audio_section_seconds = len(audio_section) / sr
            block = audio_section_seconds if audio_section_seconds < 0.5 else 0.4
            loudness = integrated_loudness(audio_section, sr, block_size=block)
            # loudness normalize audio to -16 dB LUFS
            audio_section = loudness_normalize(audio_section, loudness, -16.0)

            # keep for debugging
            # if self.debug_mode:
            #     section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)
            #     audio_test_dir = f"./tmp/audio_section_normalized"
            #     os.makedirs(audio_test_dir, exist_ok=True)
            #     sf.write(f"{audio_test_dir}/{clip_name}_{index}_{section_ts}.wav", audio_section, self.target_sample_rate)

        # keep for debugging
        # if debug_mode:
        #     os.makedirs("./tmp/audio", exist_ok=True)
        #     sf.write(
        #         f"./tmp/audio/section_{clip_name}_{index}_{seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)}.wav",
        #         audio_section, sr)

        # samples_skip_end does not skip results from being included yet
        peak_times = self._correlation_method(clip_data, audio_section=audio_section, index=index)

        # subtract sliding window seconds from peak times
        peak_times = [peak_time - subtract_seconds for peak_time in peak_times]

        # move timestamp to be before the clip
        if len(peak_times):
            peak_times_from_beginning = [time + (index * seconds_per_chunk) for time in peak_times]
            peak_times_from_beginning_new = []
            for time in peak_times_from_beginning:
                new_time = time - clip_seconds
                if new_time >= 0:
                    peak_times_from_beginning_new.append(new_time)
                else:
                    peak_times_from_beginning_new.append(0)
            peak_times_final = peak_times_from_beginning_new
        else:
            peak_times_final = []

        return peak_times_final

    # def _get_max_distance(self, downsampled_correlation_clip, downsampled_correlation_slice,):
    #     distances = np.abs(downsampled_correlation_clip-downsampled_correlation_slice)
    #     max_distance_index = np.argmax(distances)
    #     max_distance = distances[max_distance_index]
    #
    #     return max_distance,max_distance_index


    def _correlation_method(
        self,
        clip_data: ClipData,
        audio_section: NDArray[np.float32],
        index: int,
    ) -> list[float]:
        clip, clip_name, _, correlation_clip, correlation_clip_absolute_max = (
            itemgetter("clip","clip_name","sliding_window","correlation_clip","correlation_clip_absolute_max")(clip_data))

        sr = self.target_sample_rate
        debug_mode = self.debug_mode

        clip_length = len(clip)

        # zeroes_second_pad = 1
        # # pad zeros between audio and clip
        #zeroes = np.zeros(clip_length + zeroes_second_pad * sr)
        #audio = np.concatenate((audio_section, zeroes))
        # samples_skip_end = zeroes_second_pad * sr + clip_length

        # Cross-correlate and normalize correlation
        from fft_correlation import fft_correlate_1d
        # Ensure float32 and replace NaN with 0.0 (NaN comes from loudness normalization of silence)
        audio_section_f32 = np.asarray(audio_section, dtype=np.float32)
        np.nan_to_num(audio_section_f32, copy=False, nan=0.0)
        correlation = np.abs(fft_correlate_1d(audio_section_f32, clip, mode='full'))
        absolute_max = np.max(correlation)
        max_choose = max(correlation_clip_absolute_max,absolute_max)
        correlation /= max_choose

        section_ts = seconds_to_time(seconds=index * self.seconds_per_chunk, include_decimals=False)

        if debug_mode:
            import matplotlib.pyplot as plt
            print("---",file=sys.stderr)
            print(f"section_ts: {section_ts}, index {index}",file=sys.stderr)
            graph_dir = f"{self.debug_dir}/graph/cross_correlation/{clip_name}"
            os.makedirs(graph_dir, exist_ok=True)

            #Optional: plot the correlation graph to visualize
            plt.figure(figsize=(10, 4))
            plt.plot(correlation)
            plt.title('Cross-correlation between the audio clip and full track before slicing')
            plt.xlabel('Lag')
            plt.ylabel('Correlation coefficient')
            plt.savefig(
                f'{graph_dir}/{clip_name}_{index}_{section_ts}.png')
            plt.close()

        # no repetition within the duration of the clip
        distance = clip_length
        # minimum height of the peak of the correlation, don't set it too high otherwise will miss some
        # the selected ones are going to be checked for similarity
        # before adding to final peaks
        height_min = self.height_min if self.height_min is not None else 0.25
        from native_helper import find_peaks
        peaks, _ = find_peaks(correlation, height=height_min, distance=distance)

        peaks_final = []

        # for debugging
        similarities = []
        seconds = []
        #distances = []

        for peak in peaks:

            # make sure it is not out of bounds at the beginning and end after slicing
            after = peak + len(correlation_clip)//2
            before = peak - len(correlation_clip)//2
            # max_height_index = np.argmax(correlation)
            # if one_shot and peak != max_height_index:
            #     logger.warning(f"peak {peak} not equal to max_height_index {max_height_index}")
            #print("peak after",after)
            #print("len(correlation)",len(correlation))
            if after > len(correlation)+5:
                logger.warning(f"{section_ts} {clip_name} peak {peak} after is {after} > len(correlation)+5 {len(correlation)+5}, skipping")
                continue
            elif before < -5:
                logger.warning(f"{section_ts} {clip_name} peak {peak} before is {before} < -5, skipping")
                continue

            self._verify_peak_candidate(
                clip_name=clip_name,
                audio_section=audio_section,
                correlation=correlation,
                correlation_clip=correlation_clip,
                peak=peak,
                clip_length=clip_length,
                sr=sr,
                index=index,
                section_ts=section_ts,
                seconds=seconds,
                similarities=similarities,
                peaks_final=peaks_final,
            )

            if debug_mode:
                audio_test_dir = f"{self.debug_dir}/audio_section/{clip_name}"
                os.makedirs(audio_test_dir, exist_ok=True)
                # Clip to [-1.0, 1.0] to avoid clipping distortion in WAV output
                debug_audio = np.clip(audio_section[peak - len(clip):peak + len(clip)], -1.0, 1.0)
                _write_audio_file(
                    f"{audio_test_dir}/{clip_name}_{index}_{section_ts}_{peak}.wav",
                    debug_audio,
                    self.target_sample_rate
                )

        if debug_mode and len(peaks) > 0:
            peak_dir = f"{self.debug_dir}/debug/cross_correlation_{clip_name}"
            os.makedirs(peak_dir, exist_ok=True)

            print(json.dumps({"peaks": peaks, "seconds": seconds,
                              "similarities": similarities}, indent=2, cls=NumpyEncoder),
                  file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))

            print("---",file=sys.stderr)

        # convert peaks to seconds
        peak_times = [peak / sr for peak in peaks_final]

        return peak_times

    def _verify_peak_candidate(
        self,
        clip_name: str,
        audio_section: NDArray[np.float32],
        correlation: NDArray[np.float32],
        correlation_clip: NDArray[np.float32],
        peak: int,
        clip_length: int,
        sr: int,
        index: int,
        section_ts: str,
        seconds: list[float],
        similarities: list[Any],
        peaks_final: list[int],
    ) -> None:
        """Route a candidate peak to the appropriate verification method."""
        dominant_frequency = self._pure_tone_frequencies.get(clip_name)
        if dominant_frequency is not None:
            accepted = self._verify_pure_tone(
                audio_section=audio_section,
                peak=peak,
                clip_length=clip_length,
                dominant_frequency=dominant_frequency,
                sr=sr,
                section_ts=section_ts,
            )
            if accepted:
                peaks_final.append(peak)
        else:
            correlation_slice = slicing_with_zero_padding(correlation, len(correlation_clip), peak)
            correlation_slice = correlation_slice / np.max(correlation_slice)

            if len(correlation_slice) != len(correlation_clip):
                raise ValueError(f"correlation_slice length {len(correlation_slice)} not equal to correlation_clip length {len(correlation_clip)}")

            is_short_clip = clip_length / sr < SHORT_CLIP_DURATION_THRESHOLD
            self._get_peak_times_normal(
                correlation_clip=correlation_clip,
                correlation_slice=correlation_slice,
                seconds=seconds,
                peak=peak,
                clip_name=clip_name,
                index=index,
                section_ts=section_ts,
                similarities=similarities,
                peaks_final=peaks_final,
                is_short_clip=is_short_clip,
            )

    def _verify_pure_tone(
        self,
        audio_section: NDArray[np.float32],
        peak: int,
        clip_length: int,
        dominant_frequency: float,
        sr: int,
        section_ts: str,
    ) -> bool:
        """Verify a candidate peak is a pure tone with the expected frequency and duration.

        This is the RTHK beep special-case path, triggered by clip name (see
        PURE_TONE_CLIP_NAME). It exists because rthk_beep.wav does not
        cross-correlate well enough for the normal MSE+Pearson verification.

        Uses short-time spectral analysis to require a contiguous run of
        narrowband energy at the expected frequency. This rejects voiced or
        harmonic speech segments that still produce a strong correlation peak.

        Four acceptance criteria handle different signal conditions. All were
        tuned against the regression test suite in tests/test_real_data_regressions.py
        (stray clips, hourly lead-ins, hourly openings). Adjust thresholds
        there first, then re-run those tests to validate.
        """
        debug_mode = self.debug_mode
        match_start = peak - clip_length + 1
        matched_segment = extract_padded_segment(audio_section, match_start, clip_length)
        left_flank = extract_padded_segment(audio_section, match_start - clip_length, clip_length)
        right_flank = extract_padded_segment(audio_section, match_start + clip_length, clip_length)

        metrics = analyze_pure_tone_candidate(matched_segment, sr, dominant_frequency)
        left_metrics = analyze_pure_tone_candidate(left_flank, sr, dominant_frequency)
        right_metrics = analyze_pure_tone_candidate(right_flank, sr, dominant_frequency)
        max_flank_purity = max(
            left_metrics.overall_band_purity,
            right_metrics.overall_band_purity,
        )

        if not math.isclose(metrics.detected_frequency, dominant_frequency, rel_tol=0.05):
            if debug_mode:
                print(
                    f"failed pure tone check for {section_ts}: dominant "
                    f"{metrics.detected_frequency:.1f}Hz != expected "
                    f"{dominant_frequency:.1f}Hz",
                    file=sys.stderr,
                )
            return False

        # Strong, clean beep with high purity throughout. Allows slightly
        # noisy flanks since the signal itself is unambiguous.
        strict_accept = (
            metrics.overall_band_purity >= 0.80
            and metrics.longest_active_run >= 15
            and metrics.active_frame_mean_purity >= 0.84
            and max_flank_purity <= 0.15
        )

        # Weaker beep (e.g. lossy codec, lower SNR) but still clearly
        # isolated from surrounding content. Requires quieter flanks.
        isolated_accept = (
            metrics.overall_band_purity >= 0.65
            and metrics.longest_active_run >= 10
            and metrics.active_frame_mean_purity >= 0.77
            and max_flank_purity <= 0.12
        )

        # Beep with consistent frame-level activity (high active_frame_ratio)
        # but slightly lower per-frame purity — typical of beeps embedded in
        # background hiss or low-level music.
        stable_isolated_accept = (
            metrics.overall_band_purity >= 0.72
            and metrics.active_frame_ratio >= 0.85
            and metrics.longest_active_run >= 14
            and metrics.active_frame_mean_purity >= 0.76
            and max_flank_purity <= 0.12
        )

        # Very short beep (few active frames) that is highly pure where it
        # does appear. Strictest flank requirement to avoid false positives
        # from brief tonal artifacts in speech.
        short_isolated_accept = (
            metrics.overall_band_purity >= 0.72
            and metrics.active_frame_ratio >= 0.65
            and metrics.longest_active_run >= 7
            and metrics.active_frame_mean_purity >= 0.84
            and max_flank_purity <= 0.06
        )

        if not (
            strict_accept
            or isolated_accept
            or stable_isolated_accept
            or short_isolated_accept
        ):
            if debug_mode:
                print(
                    f"failed pure tone check for {section_ts}: band={metrics.overall_band_purity:.3f} "
                    f"run={metrics.longest_active_run} active={metrics.active_frame_mean_purity:.3f} "
                    f"flank=({left_metrics.overall_band_purity:.3f}, "
                    f"{right_metrics.overall_band_purity:.3f})",
                    file=sys.stderr,
                )
            return False

        if debug_mode:
            if strict_accept:
                acceptance_mode = "strict"
            elif isolated_accept:
                acceptance_mode = "isolated"
            elif stable_isolated_accept:
                acceptance_mode = "stable_isolated"
            else:
                acceptance_mode = "short_isolated"
            print(
                f"accepted pure tone {section_ts} ({acceptance_mode}): band_purity="
                f"{metrics.overall_band_purity:.3f} active_ratio="
                f"{metrics.active_frame_ratio:.3f} run="
                f"{metrics.longest_active_run} active_purity="
                f"{metrics.active_frame_mean_purity:.3f} freq="
                f"{metrics.detected_frequency:.1f}Hz flank_purity="
                f"({left_metrics.overall_band_purity:.3f}, "
                f"{right_metrics.overall_band_purity:.3f})",
                file=sys.stderr,
            )
        return True

    def _get_peak_times_normal(
        self,
        correlation_clip: NDArray[np.float32],
        correlation_slice: NDArray[np.float32],
        seconds: list[float],
        peak: int,
        clip_name: str,
        index: int,
        section_ts: str,
        similarities: list[Any],
        peaks_final: list[int],
        is_short_clip: bool = False,
    ) -> None:

        from native_helper import pearson_correlation

        debug_mode = self.debug_mode
        sr = self.target_sample_rate

        # make 10 partitions and check the middle 2 and the whole and get minimum
        # real distortions happen in the middle most of the time
        partition_count = 10
        left_bound = 4
        right_bound = 6

        partition_size = len(correlation_clip) // partition_count

        similarity_partitions = np.array([
            _mean_squared_error(correlation_clip[i * partition_size:(i + 1) * partition_size],
                                correlation_slice[i * partition_size:(i + 1) * partition_size])
            for i in range(partition_count)
        ], dtype=np.float32)

        similarity_middle = np.mean(similarity_partitions[left_bound:right_bound])
        similarity_whole = np.mean(similarity_partitions)

        if is_short_clip:
            similarity = float(similarity_whole)
        else:
            similarity = min(similarity_whole,similarity_middle)

        # Multi-window Pearson r: try different regions and pick best match
        # Downsample count scales with window width so resolution is consistent
        ds_base = 101  # for 20% window (2 partitions)
        if is_short_clip:
            pearson_windows: list[tuple[int, int, int]] = [
                (0, 10, round(ds_base * 10 / 2)),    # 0-100% → 505 samples
            ]
        else:
            pearson_windows = [
                (0, 5, round(ds_base * 5 / 2)),      # 0-50% → 252 samples
                (4, 6, ds_base),                      # 40-60% → 101 samples
                (5, 10, round(ds_base * 5 / 2)),      # 50-100% → 252 samples
            ]

        cached_clips = self._clip_cache["downsampled_pearson_windows"].get(clip_name)
        if cached_clips is None:
            cached_clips = []
            for wl, wr, ds_n in pearson_windows:
                lo = round(len(correlation_clip) * wl / partition_count)
                hi = round(len(correlation_clip) * wr / partition_count)
                cached_clips.append(resample_preserve_maxima(correlation_clip[lo:hi], ds_n))
            self._clip_cache["downsampled_pearson_windows"][clip_name] = cached_clips

        best_pearson_r = -1.0
        best_window_idx = 0
        ds_slices: list[NDArray[np.float32]] = []
        pearson_per_window: dict[str, float] = {}
        for wi, (wl, wr, ds_n) in enumerate(pearson_windows):
            lo = round(len(correlation_slice) * wl / partition_count)
            hi = round(len(correlation_slice) * wr / partition_count)
            ds_s = resample_preserve_maxima(correlation_slice[lo:hi], ds_n)
            ds_slices.append(ds_s)
            r: float = pearson_correlation(cached_clips[wi], ds_s)
            pearson_per_window[f"pearson_w{wl}_{wr}"] = r
            if r > best_pearson_r:
                best_pearson_r = r
                best_window_idx = wi

        pearson_r = best_pearson_r

        if debug_mode:
            import matplotlib.pyplot as plt
            print(f"similarity {similarity} pearson_r {pearson_r}",file=sys.stderr)
            seconds.append(peak / sr)
            self._similarity_debug[clip_name].append((index, similarity,))

            correlation_slice_graph = correlation_slice
            correlation_clip_graph = correlation_clip

            graph_max = 0.1
            if similarity <= graph_max:
                graph_dir = f"{self.debug_dir}/graph/cross_correlation_slice/{clip_name}"
                os.makedirs(graph_dir, exist_ok=True)

                plt.figure(figsize=(10, 4))
                plt.plot(correlation_slice_graph)
                plt.plot(correlation_clip_graph, alpha=0.7)
                plt.title('Cross-correlation between the audio clip and full track before slicing')
                plt.xlabel('Lag')
                plt.ylabel('Correlation coefficient')
                plt.savefig(
                    f'{graph_dir}/{clip_name}_{index}_{section_ts}_{peak}.png')
                plt.close()

                # Downsampled windows used for Pearson r
                ds_graph_dir = f"{self.debug_dir}/graph/pearson_downsampled/{clip_name}"
                os.makedirs(ds_graph_dir, exist_ok=True)

                for wi, (wl, wr, _ds_n) in enumerate(pearson_windows):
                    r_wi = pearson_per_window[f"pearson_w{wl}_{wr}"]
                    marker = " *best*" if wi == best_window_idx else ""
                    plt.figure(figsize=(10, 4))
                    plt.plot(ds_slices[wi])
                    plt.plot(cached_clips[wi], alpha=0.7)
                    plt.title(f'Partitions {wl}-{wr} (pearson_r={r_wi:.4f}){marker}')
                    plt.xlabel('Sample')
                    plt.ylabel('Correlation coefficient')
                    plt.savefig(
                        f'{ds_graph_dir}/{clip_name}_{index}_{section_ts}_{peak}_w{wl}_{wr}.png')
                    plt.close()

            best_wl, best_wr, _best_ds_n = pearson_windows[best_window_idx]
            similarities.append((similarity, {"whole": float(similarity_whole),
                                              "middle": float(similarity_middle)},
                                 {"pearson_r": pearson_r,
                                  "best_window_left": float(best_wl),
                                  "best_window_right": float(best_wr),
                                  **pearson_per_window}))

        similarity_hard_limit = 0.03
        pearson_r_threshold = 0.90

        if similarity > similarity_hard_limit:
            if debug_mode:
                print(f"failed verification for {section_ts} due to similarity {similarity} > {similarity_hard_limit}",file=sys.stderr)
        elif pearson_r >= pearson_r_threshold:
            peaks_final.append(peak)
        else:
            if debug_mode:
                print(
                    f"failed verification for {section_ts} due to similarity {similarity} pearson_r {pearson_r}",file=sys.stderr)
