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