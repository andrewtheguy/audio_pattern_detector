
# sample rate needs to be the same for both or bugs will happen
def chroma_method(clip, audio, sr, index, seconds_per_chunk, clip_name):
    hop_length = 512
    # Extract chroma features
    clip_mfcc = librosa.feature.chroma_cqt(y=clip, sr=sr, hop_length=hop_length)
    audio_mfcc = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)

    distances = []
    for i in range(audio_mfcc.shape[1] - clip_mfcc.shape[1] + 1):
        dist = np.linalg.norm(clip_mfcc - audio_mfcc[:, i:i + clip_mfcc.shape[1]])
        distances.append(dist)

    # Find minimum distance and its index
    match_index = np.argmin(distances)
    min_distance = distances[match_index]

    if debug_mode:
        section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)
        graph_dir = f"./tmp/chroma/cross_correlation_orig_{clip_name}"
        os.makedirs(graph_dir, exist_ok=True)
        # Optional: plot the correlation graph to visualize
        plt.figure(figsize=(10, 4))
        plt.plot(distances)
        plt.title('mfcc_method')
        plt.xlabel('Lag')
        plt.ylabel('y')
        plt.savefig(f'{graph_dir}/{index}_{section_ts}.png')
        plt.close()

    print(distances)

    #distances_ratio = [dist / min_distance for dist in distances]



    distances_selected = np.where(distances / min_distance <= 1.05)[0]

    # Convert match index to timestamp
    match_times = (distances_selected * hop_length) / sr  # sr is the sampling rate of audio

    return match_times

#
# # sample rate needs to be the same for both or bugs will happen
# def mfcc_method(clip, audio, sr, index, seconds_per_chunk, clip_name):
#     if(index!=0):
#         return []
#     # Extract features from the audio clip and the pattern
#     audio_features = librosa.feature.mfcc(y=audio, sr=sr)
#     pattern_features = librosa.feature.mfcc(y=clip, sr=sr)
#
#     # Compute the similarity matrix between the audio features and the pattern features
#     similarity_matrix = librosa.segment.cross_similarity(audio_features, pattern_features, mode='distance')
#     print("similarity_matrix",similarity_matrix)
#
#     # Find the indices of the maximum similarity values
#     indices = np.argmax(similarity_matrix, axis=1)
#
#
#     if debug_mode:
#         section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)
#         graph_dir = f"./tmp/mfcc_new/{clip_name}"
#         os.makedirs(graph_dir, exist_ok=True)
#         # Optional: plot the correlation graph to visualize
#         plt.figure(figsize=(10, 4))
#         plt.plot(indices)
#         plt.title('mfcc_method')
#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.savefig(f'{graph_dir}/{index}_{section_ts}.png')
#         plt.close()
#
#     print(indices)
#
#     # plt.imshow(similarity_matrix, origin='lower', aspect='auto')
#     # plt.xlabel("Frame in Long Clip")
#     # plt.ylabel("Frame in Short Clip")
#     # plt.title("Cosine Similarity Matrix")
#     # plt.show()
#     return []


# sample rate needs to be the same for both or bugs will happen
def mfcc_method2(clip, audio, sr, index, seconds_per_chunk, clip_name):
    #if index not in [54,55,56,57,58,59,60]:
    #    return []
    hop_length = 512  # Ensure this matches the hop_length used for Mel Spectrogram
    clip_len = len(clip)
    #exit(1)
    n_fft=2048
    if(clip_len < 2048):
        n_fft=clip_len
    # Extract MFCC features
    clip_mfcc = librosa.feature.mfcc(y=clip, sr=sr, hop_length=hop_length,n_fft=n_fft)
    audio_mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=hop_length,n_fft=n_fft)

    distances = []
    for i in range(audio_mfcc.shape[1] - clip_mfcc.shape[1] + 1):
        dist = np.linalg.norm(clip_mfcc - audio_mfcc[:, i:i + clip_mfcc.shape[1]])
        distances.append(dist)

    distances = np.array(distances)
    max_distance = np.max(distances)
    #print(max_distance)
    inverted_distances = (max_distance-distances)/max_distance
    #inverted_distances = inverted_distances - np.mean(inverted_distances)

    if debug_mode:
        section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)
        graph_dir = f"./tmp/mfcc/cross_correlation_orig_{clip_name}"
        os.makedirs(graph_dir, exist_ok=True)
        # Optional: plot the correlation graph to visualize
        plt.figure(figsize=(10, 4))
        plt.plot(inverted_distances)
        plt.title('mfcc_method')
        plt.xlabel('frame')
        plt.ylabel('y')
        plt.savefig(f'{graph_dir}/{index}_{section_ts}.png')
        plt.close()

    #print(distances)
    wlen = max(int(clip_len/hop_length), int(hop_length / 64))
    peaks,property = find_peaks(inverted_distances,wlen=wlen,width=[0,hop_length/64],height=0.7,prominence=0.6,rel_height=1)
    if len(peaks)>0:
        print("index",index)
        print("time", seconds_to_time(seconds=index * seconds_per_chunk))
        print("peaks",peaks)
        print("property", property)
        print("average",np.mean(inverted_distances))
        print("median", np.median(inverted_distances))
        percentile = np.percentile(inverted_distances,99)
        print("percentile", percentile)
        zscores=scipy.stats.zscore(inverted_distances)
        print("zscores", zscores[peaks])
        print("peaks values",inverted_distances[peaks])

        # Create a boolean mask based on the threshold condition
        mask = inverted_distances[peaks] - percentile >= 0
        # Apply the mask to filter the indices
        peaks = peaks[mask]

    # Convert match index to timestamp
    match_times = (peaks * hop_length) / sr  # sr is the sampling rate of audio
    if len(peaks) > 0:
        print("match_times",match_times)

    return match_times



# def advanced_correlation_method(clip, audio, sr, index, seconds_per_chunk, clip_name):
#     global plot_test_x
#     global plot_test_y
#     #if index >6:
#     #    return []
#     clip_length = len(clip)
#     clip_length_seconds = len(clip)/sr
#     #print(clip_length)
#
#
#     #threshold = 0.7  # Threshold for distinguishing peaks, need to be smaller for larger clips
#     # Cross-correlate and normalize correlation
#     correlation = correlate(audio, clip, mode='full', method='fft')
#     # if debug_mode:
#     #     section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)
#     #     graph_dir = f"./tmp/graph/cross_correlation_orig_{clip_name}"
#     #     os.makedirs(graph_dir, exist_ok=True)
#     #     # Optional: plot the correlation graph to visualize
#     #     plt.figure(figsize=(10, 4))
#     #     plt.plot(correlation)
#     #     plt.title('Cross-correlation between the audio clip and full track')
#     #     plt.xlabel('Lag')
#     #     plt.ylabel('Correlation coefficient')
#     #     plt.savefig(f'{graph_dir}/{index}_{section_ts}.png')
#     #     plt.close()
#
#     # abs
#     correlation = np.abs(correlation)
#     # replace negative values with zero in array instead of above
#     #correlation[correlation < 0] = 0
#     correlation /= np.max(correlation)
#
#     percentile=np.percentile(correlation, 95)
#     percentile_threshold = 0.05
#
#     height = 0.7
#     distance = clip_length
#     # find the peaks in the spectrogram
#     peaks, properties = find_peaks(correlation,height=height,distance=distance,prominence=0.7)
#
#     section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)
#     if debug_mode:
#         #print("clip_length", clip_length)
#         print(f"percentile for {section_ts}", percentile)
#         graph_dir = f"./tmp/graph/cross_correlation_{clip_name}"
#         os.makedirs(graph_dir, exist_ok=True)
#         # Optional: plot the correlation graph to visualize
#         plt.figure(figsize=(10, 4))
#         plt.plot(correlation)
#         plt.title('Cross-correlation between the audio clip and full track')
#         plt.xlabel('Lag')
#         plt.ylabel('Correlation coefficient')
#         plt.savefig(f'{graph_dir}/{index}_{section_ts}.png')
#         plt.close()
#
#         peak_dir = f"./tmp/peaks/cross_correlation_{clip_name}"
#         os.makedirs(peak_dir, exist_ok=True)
#         peaks_test=[]
#         for item in (peaks):
#             #plot_test_x=np.append(plot_test_x, index)
#             #plot_test_y=np.append(plot_test_y, item)
#             peaks_test.append([int(item),item/sr,correlation[item]])
#         print(json.dumps(peaks_test, indent=2), file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))
#
#
#     # won't try to match segments with too many matches
#     # because non matching looks the same as too many matches
#     if percentile >= percentile_threshold:
#         return []
#
#     max_dist = max_distance(peaks)
#
#     # multiple far away
#     if max_dist > 0:
#         max_allowed_between = clip_length + 2 * sr
#         max_allowed_between_seconds = max_allowed_between / sr
#         #print(max_dist / sr)
#         if(max_dist > clip_length+max_allowed_between):
#             print("skipping {} because it has peak with distance larger than {} seconds".format(section_ts,max_allowed_between_seconds))
#             return []
#
#     peak_times = peaks / sr
#
#     return peak_times
#
#     #correlation = downsample(int(sr/10),correlation)
#
#     # if debug:
#     #     section_ts = seconds_to_time(seconds=index * seconds_per_chunk, include_decimals=False)
#     #     graph_dir = f"./tmp/graph/cross_correlation_{clip_name}"
#     #     os.makedirs(graph_dir, exist_ok=True)
#     #     # Optional: plot the correlation graph to visualize
#     #     plt.figure(figsize=(10, 4))
#     #     plt.plot(correlation)
#     #     plt.title('Cross-correlation between the audio clip and full track')
#     #     plt.xlabel('Lag')
#     #     plt.ylabel('Correlation coefficient')
#     #     plt.savefig(f'{graph_dir}/{index}_{section_ts}.png')
#     #     plt.close()
#     #
#     #     peak_dir = f"./tmp/peaks/cross_correlation_{clip_name}"
#     #     os.makedirs(peak_dir, exist_ok=True)
#     #     print(f"np.percentile(correlation, 95) for {section_ts}",percentile)
#     #
#     #     print(json.dumps(peaks.tolist(), indent=2), file=open(f'{peak_dir}/{index}_{section_ts}.txt', 'w'))
#
#
#     # outliers = compute_mod_z_score(correlation)
#     #
#     # graph_dir = f"./tmp/graph/outliers_{clip_name}"
#     # os.makedirs(graph_dir, exist_ok=True)
#     # # Optional: plot the correlation graph to visualize
#     # plt.figure(figsize=(10, 4))
#     # plt.plot(outliers)
#     # plt.title('Cross-correlation outliers between the audio clip and full track')
#     # plt.xlabel('outliers')
#     # plt.ylabel('Correlation coefficient')
#     # plt.savefig(f'{graph_dir}/{index}_{section_ts}.png')
#     # plt.close()
#     #
#     # # peak_max = np.max(correlation)
#     # # index_max = np.argmax(correlation)
#     #
#     # max_score = np.max(outliers)
#     # max_test.append(max_score)
#     # print(f"max_score for {clip_name} {section_ts}: {max_score}")
