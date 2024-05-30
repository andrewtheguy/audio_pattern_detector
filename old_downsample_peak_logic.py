# won't work well if there are multiple occurrences of the same clip
# within the same audio_section because it inflates percentile
# and triggers multiple peaks elimination fallback
def non_repeating_correlation(clip, audio_section, sr, index, seconds_per_chunk, clip_name):
    # if clip_name == "日落大道interlude" and index not in [36,37,50,92]:
    #     return []
    # if clip_name == "日落大道smallinterlude" and index not in [13,14]:
    #     return []
    # if clip_name == "漫談法律intro" and index not in [10,11]:
    #     return []
    # if clip_name == "繼續有心人intro" and index not in [10,11]:
    #    return []
    # if clip_name == "rthk_news_report_theme" and index not in [26,27]:
    #    return []


    section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)

    clip_length = len(clip)

    #downsample_factor = int(sr / 10)

    # Cross-correlate and normalize correlation
    correlation_clip = correlate(clip, clip, mode='full', method='fft')
    #correlation_clip = downsample(correlation_clip, downsample_factor)

    # abs
    correlation_clip = np.abs(correlation_clip)
    correlation_clip /= np.max(correlation_clip)

    max_index_clip = np.argmax(correlation_clip)
    profile_clip = get_peak_profile(max_index_clip, correlation_clip)
    #bottom_ratio = get_diff_ratio(profile_clip["width_100"],profile_clip["width_75"])
    #profile_clip["bottom_ratio"] = bottom_ratio


    #correlation_clip=savgol_filter(correlation_clip, profile_clip["width_100"], 1)


    if debug_mode:
        print("clip_length", clip_length)
        #print("correlation_clip", [float(c) for c in correlation_clip])
        #raise "chafa"
        print("correlation_clip_length", len(correlation_clip))
        graph_dir = f"./tmp/graph/clip_correlation"
        os.makedirs(graph_dir, exist_ok=True)

        plt.figure(figsize=(10, 4))

        plt.plot(correlation_clip)

        plt.title('Cross-correlation for clip')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coefficient')
        plt.savefig(
            f'{graph_dir}/{clip_name}.png')
        plt.close()

        print(f"{section_ts} prominence_width_clip",profile_clip["prominence"])
        #print(f"{section_ts} bottom_ratio",profile_clip["bottom_ratio"])
        #print(f"{section_ts} left_through",left_through)
        #print(f"{section_ts} right_through",right_through)
        #print(f"{section_ts} width_clip",profile["width_100"])
        print(f"{section_ts} width_clip_whole",profile_clip["width_100"])
        print(f"{section_ts} width_clip_75",profile_clip["width_75"])
        print(f"{section_ts} width_clip_half",profile_clip["width_50"])
        #print(f"{section_ts} width_middle",width_middle)
        #print(f"{section_ts} wlen", wlen)
        peak_dir = f"./tmp/peaks_clip/non_repeating_cross_correlation_{clip_name}"
        os.makedirs(peak_dir, exist_ok=True)
        print(json.dumps({"max_index":max_index_clip,
                          "profile":profile_clip
                          }, indent=2, cls=NumpyEncoder),
              file=open(f'{peak_dir}/{clip_name}.txt', 'w'))

    print("audio_section length",len(audio_section))
    # Cross-correlate and normalize correlation
    correlation = correlate(audio_section, clip, mode='full', method='fft')
    print("correlation length", len(correlation))

    # abs
    correlation = np.abs(correlation)
    correlation /= np.max(correlation)

    max_index = np.argmax(correlation)

    padding = len(correlation_clip)/2

    beg = int(max_index-math.floor(padding))
    end = int(max_index+math.ceil(padding))

    if beg < 0:
        end = end - beg
        beg = 0

    max_index_orig = np.argmax(correlation)

    if end >= len(correlation):
        correlation = np.pad(correlation, (0, end - len(correlation)), 'constant')
    # slice
    correlation = correlation[beg:end]


    #correlation = downsample(correlation, downsample_factor)
    max_index_downsample = np.argmax(correlation)

    profile_section = get_peak_profile(max_index_downsample, correlation)

    diff_prominence_ratio = get_diff_ratio(profile_clip["prominence"],profile_section["prominence"])

    #print("correlation_clip_len_comp",len(correlation_clip), len(correlation))

    similarity = calculate_similarity(correlation_clip,correlation)

    if debug_mode:
        graph_dir = f"./tmp/graph/non_repeating_cross_correlation/{clip_name}"
        os.makedirs(graph_dir, exist_ok=True)

        #Optional: plot the correlation graph to visualize
        plt.figure(figsize=(10, 4))
        # if clip_name == "漫談法律intro" and index == 10:
        #     plt.plot(correlation[454000:454100])
        # elif clip_name == "漫談法律intro" and index == 11:
        #     plt.plot(correlation[50000:70000])
        # elif clip_name == "日落大道smallinterlude" and index == 13:
        #     plt.plot(correlation[244100:244700])
        # elif clip_name == "日落大道smallinterlude" and index == 14:
        #     plt.plot(correlation[28300:28900])
        # elif clip_name == "繼續有心人intro" and index == 10:
        #     plt.plot(correlation[440900:441000])
        # else:
        #     plt.plot(correlation)
        plt.plot(correlation)

        plt.title('Cross-correlation between the audio clip and full track before slicing')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coefficient')
        plt.savefig(
            f'{graph_dir}/{clip_name}_{index}_{section_ts}.png')
        plt.close()

        print(f"{section_ts} similarity",similarity)
        print(f"{section_ts} prominence",profile_section["prominence"])
        #print(f"{section_ts} bottom_ratio",profile_section["bottom_ratio"])
        #print(f"{section_ts} diff_bottom_ratio",diff_bottom_ratio)
        print(f"{section_ts} width_100",profile_section["width_100"])
        print(f"{section_ts} width_75",profile_section["width_75"])
        print(f"{section_ts} width_half",profile_section["width_50"])

        #print(f"{section_ts} diff_prominence_ratio",diff_prominence_ratio)
        #print(f"{section_ts} diff_width_100_ratio",diff_width_100_ratio)
        #print(f"{section_ts} diff_width_75_ratio",diff_width_75_ratio)
        #print(f"{section_ts} diff_width_50_ratio",diff_width_50_ratio)

        peak_dir = f"./tmp/peaks/non_repeating_cross_correlation_{clip_name}"
        os.makedirs(peak_dir, exist_ok=True)
        print(json.dumps({"max_index":max_index_downsample,
                          "profile":profile_section,
                          "similarity":similarity,
                          }, indent=2, cls=NumpyEncoder),
              file=open(f'{peak_dir}/{clip_name}_{index}_{section_ts}.txt', 'w'))

    qualified = True
    #if diff_prominence_ratio > 0.1:
    #    print(f"failed verification for {section_ts} due to prominence ratio {diff_prominence_ratio}")
    #    qualified = False
    if similarity > 0.002:
        print(f"failed verification for {section_ts} due to similarity {similarity}")
        qualified = False
    #if diff_width_100_ratio > 0.1:
    #    print(f"failed verification for {section_ts} due to width_100 ratio {diff_width_100_ratio}")
    #    qualified = False
    #if diff_bottom_ratio > 0.2:
    #    print(f"failed verification for {section_ts} due to width_75diff_bottom_ratio ratio {diff_bottom_ratio}")
    #    qualified = False
    #if diff_width_75_ratio > 0.5:
    #    print(f"failed verification for {section_ts} due to width_75 ratio {diff_width_75_ratio}")
    #    qualified = False
    #if diff_width_50_ratio > 0.1:
    #    print(f"failed verification for {section_ts} due to width_50 ratio {diff_width_50_ratio}")
    #    qualified = False

    if not qualified:
        print(f"failed verification for {section_ts}")
        return []
    else:
        return [max_index_orig / sr]

