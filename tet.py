import librosa
import soundfile as sf
import numpy as np

# Set the path to your M4A audio file
audio_file = "./tmp/happydaily20240416.m4a"

# Open the audio file for reading
with sf.SoundFile(audio_file) as f:
    # Get the sample rate of the audio file
    sample_rate = f.samplerate

    # Set the hop length (number of samples between each analysis frame)
    hop_length = 512

    # Create a buffer to store the audio data
    buffer = []

    # Read the audio data in chunks
    for i in range(0, len(f), hop_length):
        # Read a chunk of audio data
        chunk = f.read(hop_length)

        # Append the chunk to the buffer
        buffer.append(chunk)

        # Convert the buffer to a numpy array
        audio_data = np.concatenate(buffer)

        # Perform any desired audio processing or analysis here
        # For example, you can compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)

        # Do something with the processed audio data
        # For example, you can display the mel spectrogram
        librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

        # Clear the buffer for the next iteration
        buffer = []