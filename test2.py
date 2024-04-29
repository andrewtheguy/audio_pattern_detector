import librosa
from dtw import dtw

pattern, _ = librosa.load(librosa.ex('pistachio'))
y, sr = librosa.load(librosa.ex('pistachio'), offset=10)


# Calculate onset strength for both the pattern and the main audio file
onset_env_y = librosa.onset.onset_strength(y=y, sr=sr)
onset_env_pattern = librosa.onset.onset_strength(y=pattern, sr=sr)

# Use Dynamic Time Warping to align the two onset strength sequences
alignment = dtw(onset_env_y, onset_env_pattern, keep_internals=True)

# Print the normalized index of pattern in y
#print('Normalized index: ', alignment.normalizedIndex[0])
## Display the warping curve, i.e. the alignment curve
plot=alignment.plot(type="threeway")
