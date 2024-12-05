from collections import defaultdict
import json
import logging
import os

from operator import itemgetter
from pathlib import Path

import numpy as np

from scipy.signal import correlate
import math
import matplotlib.pyplot as plt

import pyloudnorm as pyln

import warnings
from andrew_utils import seconds_to_time

from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error

import soundfile as sf

from audio_pattern_detector.numpy_encoder import NumpyEncoder
from audio_pattern_detector.audio_utils import slicing_with_zero_padding, load_audio_file, convert_audio_arr_to_float, \
    downsample_preserve_maxima, ffmpeg_get_16bit_pcm
from audio_pattern_detector.detection_utils import area_of_overlap_ratio, is_pure_tone

logger = logging.getLogger(__name__)

#ignore possible clipping
warnings.filterwarnings('ignore', module='pyloudnorm')

class AudioPatternDetector:

    def __init__(self, clip_paths, debug_mode=False):
        self.clip_paths = clip_paths
        self.debug_mode = debug_mode
        #self.correlation_cache_correlation_method = {}
        self.normalize = True
        self.target_sample_rate = 8000

        for clip_path in clip_paths:
            if not os.path.exists(clip_path):
                raise ValueError(f"Clip {clip_path} does not exist")

    # could cause issues with small overlap when intro is followed right by news report
    def find_clip_in_audio(self, full_audio_path):
        clip_paths = self.clip_paths
        #self.correlation_cache_correlation_method.clear()

        if not os.path.exists(full_audio_path):
            raise ValueError(f"Full audio {full_audio_path} does not exist")

        seconds_per_chunk = 60

        # 2 bytes per channel on every sample for 16 bits (int16)
        # times two because it is (int16, mono)
        chunk_size = (seconds_per_chunk * self.target_sample_rate) * 2

        # Initialize parameters

        previous_chunk = None  # Buffer to maintain continuity between chunks

        all_peak_times = {clip_path: [] for clip_path in clip_paths}


        with ffmpeg_get_16bit_pcm(full_audio_path, self.target_sample_rate, ac=1) as stdout:

            i = 0

            clip_datas={}
            clip_cache={
                "downsampled_correlation_clips":{},
                "is_pure_tone_pattern":{},
                "similarity_debug":defaultdict(list),
            }

            clips_already = set()

            for clip_path in clip_paths:
                # Load the audio clip
                clip = load_audio_file(clip_path, sr=self.target_sample_rate)
                # convert to float
                clip = convert_audio_arr_to_float(clip)

                clip_name, _ = os.path.splitext(os.path.basename(clip_path))

                if clip_name in clips_already:
                    raise ValueError(f"clip {clip_name} needs to be unique")

                clips_already.add(clip_name)

                clip_seconds = len(clip) / self.target_sample_rate

                sliding_window = self._get_chunking_timing_info(clip_name,clip_seconds,seconds_per_chunk)

                if self.normalize:
                    # max_loudness = np.max(np.abs(clip))
                    # clip = clip / max_loudness
                    sr = self.target_sample_rate
                    #clip_second = clip_length / sr

                    # normalize loudness
                    if clip_seconds < 0.5:
                        meter = pyln.Meter(sr, block_size=clip_seconds)
                    else:
                        meter = pyln.Meter(sr)  # create BS.1770 meter
                    loudness = meter.integrated_loudness(clip)

                    # loudness normalize audio to -16 dB LUFS
                    clip = pyln.normalize.loudness(clip, loudness, -16.0)

                correlation_clip,absolute_max = self._get_clip_correlation(clip)

                if self.debug_mode:
                    print(f"clip_length {clip_name}", len(clip))
                    print(f"clip_length {clip_name} seconds", len(clip)/self.target_sample_rate)
                    print("correlation_clip_length", len(correlation_clip))
                    graph_dir = f"../tmp/graph/clip_correlation"
                    os.makedirs(graph_dir, exist_ok=True)

                    plt.figure(figsize=(10, 4))

                    plt.plot(correlation_clip)
                    plt.title('Cross-correlation of the audio clip itself')
                    plt.xlabel('Lag')
                    plt.ylabel('Correlation coefficient')
                    plt.savefig(
                        f'{graph_dir}/{clip_name}.png')
                    plt.close()

                clip_datas[clip_path] = {"clip":clip,
                                         "clip_name":clip_name,
                                         "sliding_window":sliding_window,
                                         "correlation_clip":correlation_clip,
                                         "correlation_clip_absolute_max":absolute_max,
                                         #"downsampled_correlation_clip":downsampled_correlation_clip,
                                         }

            # Process audio in chunks
            while True:
                in_bytes = stdout.read(chunk_size)
                if not in_bytes:
                    break
                # Convert bytes to numpy array
                chunk = np.frombuffer(in_bytes, dtype="int16")
                # convert to float
                chunk = convert_audio_arr_to_float(chunk)

                for clip_path in clip_paths:
                    clip_data = clip_datas[clip_path]


                    peak_times = self._process_chunk(chunk=chunk,
                                                     sr=self.target_sample_rate,
                                                     previous_chunk=previous_chunk,
                                                     index=i,
                                                     clip_data=clip_data,
                                                     clip_cache=clip_cache,
                                                     seconds_per_chunk=seconds_per_chunk,
                                                     )

                    all_peak_times[clip_path].extend(peak_times)

                # Update previous_chunk to current chunk
                previous_chunk = chunk
                i = i + 1

            if self.debug_mode:

                suffix = Path(full_audio_path).stem

                for clip_path in clip_paths:
                    clip_name, _ = os.path.splitext(os.path.basename(clip_path))

                    # similarity debug
                    graph_dir = f"./tmp/graph/mean_squared_error_similarity/{clip_name}"
                    os.makedirs(graph_dir, exist_ok=True)

                    x_coords = []
                    y_coords = []

                    for index,similarity in clip_cache['similarity_debug'][clip_name]:
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


        return all_peak_times

    def _get_chunking_timing_info(self, clip_name, clip_seconds, seconds_per_chunk):
        sliding_window = math.ceil(clip_seconds)

        if (sliding_window != clip_seconds):
            print(f"adjusted sliding_window from {clip_seconds} to {sliding_window} for {clip_name}")
        # sliding_window = 5
        #
        # if (sliding_window < clip_seconds + 5):
        #     # need to extend the sliding window to overlap the clip
        #     sliding_window = clip_seconds + 5
        #     print(f"adjusted sliding_window to {sliding_window} for {clip_name}")

        # this should not happen anyways because the seconds per chunk is too small
        if (seconds_per_chunk < sliding_window * 2):
            seconds_per_chunk = sliding_window * 10
            raise ValueError(f"seconds_per_chunk {seconds_per_chunk} is too small")

        return sliding_window

    def _get_clip_correlation(self, clip):
        # Cross-correlate and normalize correlation
        correlation_clip = correlate(clip, clip, mode='full', method='fft')

        # abs
        correlation_clip = np.abs(correlation_clip)
        absolute_max = np.max(correlation_clip)
        correlation_clip /= absolute_max

        return correlation_clip,absolute_max


    # sliding_window: for previous_chunk in seconds from end
    # index: for debugging by saving a file for audio_section
    # seconds_per_chunk: default seconds_per_chunk
    def _process_chunk(self, chunk, clip_data, clip_cache, sr, previous_chunk, index, seconds_per_chunk):
        clip, clip_name, sliding_window = itemgetter("clip","clip_name","sliding_window")(clip_data)
        clip_seconds = len(clip) / sr
        chunk_seconds = len(chunk) / sr
        # Concatenate previous chunk for continuity in processing
        if previous_chunk is not None:
            if chunk_seconds < seconds_per_chunk:  # too small
                # no need for sliding window since it is the last piece
                subtract_seconds = -(chunk_seconds - seconds_per_chunk)
                audio_section_temp = np.concatenate((previous_chunk, chunk))[(-seconds_per_chunk * sr):]
                audio_section = np.concatenate((audio_section_temp, np.array([])))
            else:
                subtract_seconds = sliding_window
                audio_section = np.concatenate((previous_chunk[int(-sliding_window * sr):], chunk, np.array([])))
        else:
            subtract_seconds = 0
            audio_section = np.concatenate((chunk, np.array([])))

        if self.normalize:
            #max_loudness = np.max(np.abs(audio_section))
            #audio_section = audio_section / max_loudness
            audio_section_seconds = len(audio_section) / sr
            #normalize loudness
            if audio_section_seconds < 0.5:
                # not sure if there are valid use cases for this
                #raise ValueError("audio_section_seconds < 0.5 second")
                meter = pyln.Meter(sr, block_size=audio_section_seconds)
            else:
                meter = pyln.Meter(sr)  # create BS.1770 meter

            loudness = meter.integrated_loudness(audio_section)

            # loudness normalize audio to -16 dB LUFS
            audio_section = pyln.normalize.loudness(audio_section, loudness, -16.0)

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
        peak_times = self._correlation_method(clip_data, audio_section=audio_section, sr=sr, index=index,
                                              seconds_per_chunk=seconds_per_chunk,
                                              clip_cache=clip_cache,
                                              )

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


    # won't work well for very short clips like single beep
    # because it is more likely to have false positives or miss good ones
    def _correlation_method(self, clip_data, clip_cache, audio_section, sr, index, seconds_per_chunk):
        clip, clip_name, sliding_window, correlation_clip, correlation_clip_absolute_max= (
            itemgetter("clip","clip_name","sliding_window","correlation_clip","correlation_clip_absolute_max")(clip_data))

        if clip_cache["is_pure_tone_pattern"].get(clip_name) is None:
            clip_cache["is_pure_tone_pattern"][clip_name] = is_pure_tone(clip, sr)

        is_pure_tone_pattern = clip_cache["is_pure_tone_pattern"][clip_name]

        debug_mode = self.debug_mode

        clip_length = len(clip)
        clip_length_seconds = clip_length / sr

        very_short_clip = clip_length_seconds < 0.5

        if very_short_clip and not is_pure_tone_pattern:
            raise ValueError(f"very short clip {clip_name} is not supported yet unless it is pure tone pattern, it has {clip_length_seconds} seconds")

        # zeroes_second_pad = 1
        # # pad zeros between audio and clip
        #zeroes = np.zeros(clip_length + zeroes_second_pad * sr)
        #audio = np.concatenate((audio_section, zeroes))
        # samples_skip_end = zeroes_second_pad * sr + clip_length

        # Cross-correlate and normalize correlation
        correlation = correlate(audio_section, clip, mode='full', method='fft')
        # abs
        correlation = np.abs(correlation)
        absolute_max = np.max(correlation)
        max_choose = max(correlation_clip_absolute_max,absolute_max)
        correlation /= max_choose

        section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)

        if debug_mode:
            print(f"---")
            print(f"section_ts: {section_ts}, index {index}")
            graph_dir = f"./tmp/graph/cross_correlation/{clip_name}"
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
        height_min = 0.25
        peaks, _ = find_peaks(correlation, height=height_min, distance=distance)

        peaks_final = []

        # for debugging
        area_props = []
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

            # slice with the center on the peak
            correlation_slice = slicing_with_zero_padding(correlation, len(correlation_clip), peak)
            correlation_slice = correlation_slice/np.max(correlation_slice)

            if len(correlation_slice) != len(correlation_clip):
                raise ValueError(f"correlation_slice length {len(correlation_slice)} not equal to correlation_clip length {len(correlation_clip)}")

            if is_pure_tone_pattern:
                self._get_peak_times_beep_v3(correlation_clip=correlation_clip,
                                          correlation_slice=correlation_slice,
                                          seconds=seconds,
                                          peak=peak,
                                          clip_name=clip_name,
                                          index=index,
                                          section_ts=section_ts,
                                          similarities=similarities,
                                          peaks_final=peaks_final,
                                          clip_cache=clip_cache,
                                          area_props=area_props)
                # self._get_peak_times_beep_v2(
                #                                  audio=audio[peak - len(clip):peak + len(clip)],
                #                                  peak=peak,
                #                                  peaks_final=peaks_final,
                #                                  clip_cache=clip_cache,
                #                                  area_props=area_props,
                #                                  clip_name=clip_name,
                #                                  index=index,
                #                                  section_ts=section_ts,)
            else:
                self._get_peak_times_normal(correlation_clip=correlation_clip,
                                                correlation_slice=correlation_slice,
                                                seconds=seconds,
                                                peak=peak,
                                                clip_name=clip_name,
                                                index=index,
                                                section_ts=section_ts,
                                                similarities=similarities,
                                                peaks_final=peaks_final,
                                                area_props=area_props,
                                                clip_cache=clip_cache)

            if debug_mode:
                audio_test_dir = f"./tmp/audio_section/{clip_name}"
                os.makedirs(audio_test_dir, exist_ok=True)
                sf.write(f"{audio_test_dir}/{clip_name}_{index}_{section_ts}_{peak}.wav", audio_section[peak - len(clip):peak + len(clip)],
                         self.target_sample_rate)

        if debug_mode and len(peaks) > 0:
            peak_dir = f"./tmp/debug/cross_correlation_{clip_name}"
            os.makedirs(peak_dir, exist_ok=True)

            print(json.dumps({"peaks": peaks, "seconds": seconds,
                              "area_props":area_props,
                              #"distances": distances,
                              "similarities": similarities}, indent=2, cls=NumpyEncoder),
                  file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))

            print(f"---")

        # convert peaks to seconds
        peak_times = [peak / sr for peak in peaks_final]

        return peak_times

    def _get_peak_times_normal(self, correlation_clip, correlation_slice, seconds, peak, clip_name, index,
                             section_ts, similarities, peaks_final, clip_cache, area_props):

        debug_mode = self.debug_mode
        sr = self.target_sample_rate

        # make 10 partitions and check the middle 2 and the whole and get minimum
        # real distortions happen in the middle most of the time
        partition_count = 10
        left_bound = 4
        right_bound = 6

        partition_size = len(correlation_clip) // partition_count

        similarity_partitions = []
        for i in range(partition_count):
            similarity_partitions.append(
                mean_squared_error(correlation_clip[i * partition_size:(i + 1) * partition_size],
                                   correlation_slice[i * partition_size:(i + 1) * partition_size]))

        similarity_middle = np.mean(similarity_partitions[left_bound:right_bound])
        similarity_whole = np.mean(similarity_partitions)
        #similarity_whole = 0
        similarity_left = 0
        similarity_right = 0
        # similarity_left = np.mean(similarity_partitions[0:5])
        # similarity_right = np.mean(similarity_partitions[5:10])

        #similarity = similarity_middle
        similarity = min(similarity_whole,similarity_middle)

        # similarity = min(similarity_left,similarity_middle,similarity_right)
        # similarity = similarity_whole = (similarity_left + similarity_right)/2

        lower_limit = round(len(correlation_clip) * left_bound / partition_count)
        upper_limit = round(len(correlation_clip) * right_bound / partition_count)
        area_prop = area_of_overlap_ratio(correlation_clip[lower_limit:upper_limit],
                                                                              correlation_slice[
                                                                              lower_limit:upper_limit])

        # ratio of difference between overlapping area and non-overlapping area
        # needed to check when mean_squared_error is high enough
        diff_overlap_ratio = area_prop["diff_overlap_ratio"]

        if debug_mode:
            similarity_debug = clip_cache["similarity_debug"]
            print("similarity", similarity)
            seconds.append(peak / sr)
            similarity_debug[clip_name].append((index, similarity,))

            correlation_slice_graph = correlation_slice
            correlation_clip_graph = correlation_clip

            graph_max = 0.1
            if similarity <= graph_max:
                graph_dir = f"./tmp/graph/cross_correlation_slice/{clip_name}"
                os.makedirs(graph_dir, exist_ok=True)

                # Optional: plot the correlation graph to visualize
                plt.figure(figsize=(10, 4))
                plt.plot(correlation_slice_graph)
                plt.plot(correlation_clip_graph, alpha=0.7)
                plt.title('Cross-correlation between the audio clip and full track before slicing')
                plt.xlabel('Lag')
                plt.ylabel('Correlation coefficient')
                plt.savefig(
                    f'{graph_dir}/{clip_name}_{index}_{section_ts}_{peak}.png')
                plt.close()

            area_props.append([diff_overlap_ratio, area_prop])

            similarities.append((similarity, {"whole": similarity_whole,
                                              "left": similarity_left,
                                              "middle": similarity_middle,
                                              "right": similarity_right,
                                              "left_right_diff": abs(similarity_left - similarity_right),
                                              }))

        similarity_threshold = 0.01
        similarity_threshold_check_area = 0.002

        # reject if similarity is high enough and little area overlap
        diff_overlap_ratio_threshold = 0.5

        if similarity_threshold <= similarity_threshold_check_area:
            raise ValueError(
                f"similarity_threshold {similarity_threshold} needs to be larger than similarity_threshold_check_area {similarity_threshold_check_area}")

        if similarity > similarity_threshold:
            if debug_mode:
                print(f"failed verification for {section_ts} due to similarity {similarity} > {similarity_threshold}")
        # if similarity is between similarity_threshold and similarity_threshold_check_area, check shape ratio
        elif similarity > similarity_threshold_check_area and diff_overlap_ratio > diff_overlap_ratio_threshold:
            if debug_mode:
                print(
                    f"failed verification for {section_ts} due to diff_overlap_ratio {diff_overlap_ratio} > {diff_overlap_ratio_threshold}")
        else:  # if similarity is less than similarity_threshold_check_area, no need to check area ratio
            peaks_final.append(peak)

    # # doesn't work well
    # def _get_peak_times_beep_v2(self,audio,peak,peaks_final,clip_cache,area_props,clip_name,index,section_ts):
    #
    #     sr = self.target_sample_rate
    #     debug_mode = self.debug_mode
    #
    #     result = is_news_report_beep(audio, sr,f"{clip_name}_{index}_{section_ts}_{peak}")
    #     detected = result['is_news_report_clip']
    #
    #     if debug_mode:
    #         print("detected", detected)
    #         area_props.append({"detect_sine_tone_result": result})
    #         audio_test_dir = f"./tmp/clip_audio_news_beep"
    #         os.makedirs(audio_test_dir, exist_ok=True)
    #         sf.write(f"{audio_test_dir}/{clip_name}_{index}_{section_ts}_{peak}.wav", audio, self.target_sample_rate)
    #
    #     if detected:
    #         peaks_final.append(peak)

    # # no longer in use, it sometimes misses those in between beeps
    # def _get_peak_times_beep_v1(self,correlation_clip,correlation_slice,seconds,peak,clip_name,index,section_ts,similarities,peaks_final,clip_cache,area_props):
    #     # short beep is very sensitive, it is better to miss some than to have false positives
    #     similarity_threshold = 0.002
    #
    #     sr = self.target_sample_rate
    #     debug_mode = self.debug_mode
    #
    #     beep_target_num_sample_after_resample = 101
    #
    #     downsampled_correlation_clip = clip_cache["downsampled_correlation_clips"].get(clip_name)
    #
    #     if downsampled_correlation_clip is None:
    #         downsampled_correlation_clip = downsample_preserve_maxima(correlation_clip, beep_target_num_sample_after_resample)
    #         clip_cache["downsampled_correlation_clips"][clip_name] = downsampled_correlation_clip
    #
    #     downsampled_correlation_slice = downsample_preserve_maxima(correlation_slice, beep_target_num_sample_after_resample)
    #
    #     correlation_clip = downsampled_correlation_clip
    #     correlation_slice = downsampled_correlation_slice
    #
    #     similarity = mean_squared_error(correlation_clip, correlation_slice)
    #
    #     similarity_whole = similarity
    #
    #     if debug_mode:
    #         print("similarity", similarity)
    #         seconds.append(peak / sr)
    #         similarity_debug = clip_cache["similarity_debug"]
    #         similarity_debug[clip_name].append((index, similarity,))
    #
    #         correlation_slice_graph = correlation_slice
    #         correlation_clip_graph = correlation_clip
    #
    #         graph_max = 0.1
    #         if similarity <= graph_max:
    #             graph_dir = f"./tmp/graph/cross_correlation_slice/{clip_name}"
    #             os.makedirs(graph_dir, exist_ok=True)
    #
    #             # Optional: plot the correlation graph to visualize
    #             plt.figure(figsize=(10, 4))
    #             plt.plot(correlation_slice_graph)
    #             plt.plot(correlation_clip_graph, alpha=0.7)
    #             plt.title('Cross-correlation between the audio clip and full track before slicing')
    #             plt.xlabel('Lag')
    #             plt.ylabel('Correlation coefficient')
    #             plt.savefig(
    #                 f'{graph_dir}/{clip_name}_{index}_{section_ts}_{peak}.png')
    #             plt.close()
    #
    #         similarities.append((similarity, {"whole": similarity_whole,
    #                                           "left": 0,
    #                                           "middle": 0,
    #                                           "right": 0,
    #                                           "left_right_diff": 0,
    #                                           }))
    #
    #         if similarity > similarity_threshold:
    #             if debug_mode:
    #                 print(
    #                     f"failed verification for {section_ts} due to similarity {similarity} > {similarity_threshold}")
    #         else:
    #             peaks_final.append(peak)

    # matching pattern should overlap almost completely with beep pattern, unless they are too dissimilar
    def _get_peak_times_beep_v3(self,correlation_clip,correlation_slice,seconds,peak,clip_name,index,section_ts,similarities,peaks_final,clip_cache,area_props):

        sr = self.target_sample_rate
        debug_mode = self.debug_mode

        beep_target_num_sample_after_resample = 101

        downsampled_correlation_clip = clip_cache["downsampled_correlation_clips"].get(clip_name)

        if downsampled_correlation_clip is None:
            downsampled_correlation_clip = downsample_preserve_maxima(correlation_clip, beep_target_num_sample_after_resample)
            clip_cache["downsampled_correlation_clips"][clip_name] = downsampled_correlation_clip

        downsampled_correlation_slice = downsample_preserve_maxima(correlation_slice, beep_target_num_sample_after_resample)

        correlation_clip = downsampled_correlation_clip
        correlation_slice = downsampled_correlation_slice

        area_prop = area_of_overlap_ratio(correlation_clip,correlation_slice)

        overlap_ratio = area_prop["overlapping_area"]/area_prop["area_control"]

        similarity = mean_squared_error(correlation_clip, correlation_slice)

        similarity_whole = similarity

        if debug_mode:
            print("similarity", similarity)
            seconds.append(peak / sr)
            similarity_debug = clip_cache["similarity_debug"]
            similarity_debug[clip_name].append((index, similarity,))

            correlation_slice_graph = correlation_slice
            correlation_clip_graph = correlation_clip

            graph_max = 0.1
            if similarity <= graph_max:
                graph_dir = f"./tmp/graph/cross_correlation_slice/{clip_name}"
                os.makedirs(graph_dir, exist_ok=True)

                # Optional: plot the correlation graph to visualize
                plt.figure(figsize=(10, 4))
                plt.plot(correlation_slice_graph)
                plt.plot(correlation_clip_graph, alpha=0.7)
                plt.title('Cross-correlation between the audio clip and full track before slicing')
                plt.xlabel('Lag')
                plt.ylabel('Correlation coefficient')
                plt.savefig(
                    f'{graph_dir}/{clip_name}_{index}_{section_ts}_{peak}.png')
                plt.close()

            similarities.append((similarity, {"whole": similarity_whole,
                                              "left": 0,
                                              "middle": 0,
                                              "right": 0,
                                              "left_right_diff": 0,
                                              }))
            area_props.append([overlap_ratio, area_prop])


        similarity_threshold = 0.01

        similarity_threshold_check_area_upper = 0.003

        similarity_threshold_check_area = 0.002

        if similarity > similarity_threshold:
            if debug_mode:
                print(f"failed verification for {section_ts} due to similarity {similarity} > {similarity_threshold}")
        # if similarity is between similarity_threshold and similarity_threshold_check_area, check shape ratio
        elif similarity > similarity_threshold_check_area_upper and overlap_ratio < 0.99:
            if debug_mode:
                print(
                    f"failed verification for {section_ts} due to similarity {similarity} overlap_ratio {overlap_ratio} < 0.99")
        # similar enough, lower area ratio threshold
        elif similarity > similarity_threshold_check_area and overlap_ratio < 0.98:
            if debug_mode:
                print(
                    f"failed verification for {section_ts} due to similarity {similarity} overlap_ratio {overlap_ratio} < 0.98")
        else:
            if debug_mode:
                print(
                    f"accepted {section_ts} with similarity {similarity} and overlap_ratio {overlap_ratio}")
            peaks_final.append(peak)