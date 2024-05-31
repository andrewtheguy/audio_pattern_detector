## usage
```shell
# detect pattern from audio file with debug
python match.py --pattern-file ./audio_clips/happydailyfemale2.wav --audio-file file --match-method correlation

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

### default cross-correlation
Works well with repeating or non-repeating patterns that are loud enough within the audio section because it adds the normalized clip
at the end of the audio section, which helps to eliminate false positives that are relatively softer than the clip,
which happens often.
won't work well for patterns that are too short or not loud enough within the audio section. 

### non-repeating cross-correlation
Works well for a pattern that is guaranteed not to be repeating within the time frame and again that is long enough,
and it can still detect pattern that is not prominent enough, and it is overall more accurate than the default cross-correlation,
it won't work well for repeating patterns because it only picks the loudest or the best match instead of going through all potential matches.
While it works slightly better than the default cross-correlation for very short clips, it still returns false positives for these when the short fake one is very similar to the real one.

### limitations
overall, both method don't work well for patterns that are too short because of too many false positives, and patterns that are not loud enough are more likely to fail,
especially with the default method that allows repeats.

## testing
use pytest to test because not all of them are written using default python unittest module, and pytest is more flexible and easier to use.