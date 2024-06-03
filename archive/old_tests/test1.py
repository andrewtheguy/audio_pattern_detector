
if __name__ == "__main__":
    import librosa
    import numpy as np

    from utils import extract_prefix
    from sklearn.metrics.pairwise import cosine_similarity

    print(extract_prefix("testagain2022041"))  # None
    print(extract_prefix("testagain20220414"))  # returns ('testagain', '20220414')
    print(extract_prefix("testagain220220414"))  # returns ('testagain2', '20220414')
    print(extract_prefix("testagain220220414hahanada"))  # returns ('testagain2', '20220414')

    hop_length = 512
    y_ref, sr = librosa.load(librosa.ex('pistachio'))
    y_comp, sr = librosa.load(librosa.ex('pistachio'), offset=10)
    #chroma_ref = librosa.feature.chroma_cqt(y=y_ref, sr=sr, hop_length=hop_length)
    #chroma_comp = librosa.feature.chroma_cqt(y=y_comp, sr=sr, hop_length=hop_length)
    # Use time-delay embedding to get a cleaner recurrence matrix
    #x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
    #x_comp = librosa.feature.stack_memory(chroma_comp, n_steps=10, delay=3)
    #xsim = librosa.segment.cross_similarity(x_comp, x_ref)

    # Extract features from the audio clip and the pattern
    audio_features = librosa.feature.mfcc(y=y_comp, sr=sr,hop_length=hop_length)
    pattern_features = librosa.feature.mfcc(y=y_ref, sr=sr,hop_length=hop_length)

    # Compute the similarity matrix between the audio features and the pattern features
    #similarity_matrix = librosa.segment.cross_similarity(audio_features, pattern_features)

    #print(similarity_matrix,file=sys.stderr)

    #indices = np.argmax(similarity_matrix, axis=1)
    #print(sorted(indices*hop_length/sr))

    similarity = cosine_similarity(audio_features.T, pattern_features.T)
    #print("Cosine Similarity: ", similarity)

    indices = np.argmax(similarity, axis=1)
    print(sorted(indices*hop_length/sr))