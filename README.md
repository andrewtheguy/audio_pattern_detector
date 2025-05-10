# Audio Pattern Detector
This is a project that detects audio patterns specified by audio clips on a target audio file. 
It is designed to detect audio patterns like intros, breaks, and outros from prerecorded radio shows and podcasts,
which can be helpful to get the sections of the particular radio show or podcast that you are interested in.
It also helps for openai whisper transcription preprocessing by trimming the audio to the section that you are interested in.
It uses cross-correlation to detect the potential matching pattern, and then uses mean square error and overlapping areas on cross correlation graph to eliminate false positives.


## usage
```shell
# detect pattern from audio file, add --no-debug for disable debugging
python match.py --audio-file ./sample_audios/audio_section.wav --pattern-file ./sample_audios/clips/dada.wav

# convert audio file to target sample rate
python convert.py --audio-file ./tmp/dada.wav --dest-file ./sample_audios/clips/dada.wav

```

## audio pattern detection methods
currently only supports cross-correlation

### default cross-correlation
Picks all peaks that are above a certain threshold, and then eliminate false positives with cross similarity and mean square error.
Works well with repeating or non-repeating patterns that are loud enough within the audio section because it adds the normalized clip
at the end of the audio section, which helps to eliminate false positives that are much softer or non-related to the clip.
won't work well for patterns that are too short, currently it disallow short clips unless it is pure tone pattern, if it is short and pure tone pattern, then a special correlation logic is used to match. 

It will miss distorted patterns like this because error score is too high and area overlap ratio is too low:
![rthk_beep_39_00:39:00_478782](https://github.com/user-attachments/assets/80669708-b8f9-461c-ae6c-2edddb161904)



## testing
use pytest to test because not all of them are written using default python unittest module, and pytest is more flexible and easier to use.

## podcast publishing
it publishes the media to free ipfs hosting, then it uploads the xml feed to a free cloudflare worker through an external custom endpoint https://github.com/andrewtheguy/podcast_hosting that serves the feed with ipfs urls.
