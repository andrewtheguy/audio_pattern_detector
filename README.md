# Audio Pattern Detection
This is a project that detects audio patterns specified by audio clips on a target audio file. It is designed to detect consistent patterns from prerecorded radio shows and podcasts.
It uses cross-correlation to detect the potential matching pattern, and then uses mean square error and overlapping areas on cross correlation graph to eliminate false positives.

## usage
```shell
# detect pattern from audio file with debug
python match.py --audio-file /Volumes/andrewdata/ftp/grabradiostreamed/am1430/multiple/日落大道/日落大道20240523_1600_s_1.m4a --pattern-file ./audio_clips/am1430/日落大道interlude.wav

# schedule periodic scraping from rthk
python schedule.py

# convert audio file to target sample rate
python convert.py --pattern-file  /Volumes/andrewdata/audio_test/knowledge_co_e_word_intro.wav --dest-file audio_clips/knowledge_co_e_word_intro.wav

# scrape audio from downloaded rthk audio
python scrape_rthk.py scrape --audio-file /Volumes/andrewdata/ftp/rthk/original/morningsuite/morningsuite20240425.m4a

# scrape audio from downloaded am1430 audio
python scrape_standard.py --audio-file /Volumes/andrewdata/ftp/grabradiostreamed/am1430/multiple/受之有道/受之有道20240509_1800_s_1.m4a

```

## audio pattern detection methods
currently only supports cross-correlation

### default cross-correlation
Picks all peaks that are above a certain threshold, and then eliminate false positives with cross similarity and mean square error.
Works well with repeating or non-repeating patterns that are loud enough within the audio section because it adds the normalized clip
at the end of the audio section, which helps to eliminate false positives that are much softer or non-related to the clip.
won't work well for patterns that are too short or not loud enough within the audio section. 

## testing
use pytest to test because not all of them are written using default python unittest module, and pytest is more flexible and easier to use.

## podcast publishing
it publishes the media to free ipfs hosting, then it uploads the xml feed to a free cloudflare worker through an external custom endpoint https://github.com/andrewtheguy/podcast_hosting that serves the feed with ipfs urls.