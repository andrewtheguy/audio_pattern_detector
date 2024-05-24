## audio pattern detection methods

### default cross-correlation
works for repeating patterns that are not too short in an audio section and are prominent enough, it won't work well
for very short beep or patterns that are not loud enough in the audio section. 

### non-repeating cross-correlation
Works well for a pattern that is guaranteed not to be repeating within the time frame and again that is long enough,
and it can still detect pattern that is not prominent enough, and it is overall more accurate than the default cross-correlation,
it won't work well for repeating patterns because it uses statistical methods to eliminate potential false positives such as nth-percentile
being too high or multiple patterns within the same time frame.
While it works slightly better than the default cross-correlation for very short clips, it still returns false positives for these when
there are exactly one false positive found in the audio section and it is not possible to distinguish without listening because the peak
looks about the same as the real one.