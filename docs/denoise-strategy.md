# Pattern Clip Strategy

## Preferred: extract clips from the source encoding

The best pattern clips are extracted directly from the same encoding as the audio being searched. For example, if the target audio is an Opus HLS stream, extract the clip from an Opus recording rather than from a lossless or differently-encoded source.

Lossy codecs (Opus, AAC) alter the cross-correlation shape. A clip extracted from the same codec shares the same artifacts, so the correlation envelope stays tight and the Pearson r stays high. A clip from a different encoding (e.g. a clean WAV clip matched against Opus audio) produces an inflated noise floor in the cross-correlation, which deflates the Pearson r and can cause missed detections.

```bash
# Extract a clip from an Opus source at the detection timestamp
ffmpeg -i source.ogg -ss <timestamp> -t <duration> -ar 8000 -ac 1 -y clip.wav
```

Use the old (pre-change) detector or a lower `--height-min` to find the timestamp in the source audio, then extract the clip at that position.

## Fallback: denoise strategy

When a source-matched clip is not available (e.g. the pattern only exists in a clean recording), denoising the clip can improve results. Background noise in the pattern correlates with unrelated parts of the audio stream, widening the cross-correlation envelope and reducing the Pearson r score.

Denoising helps in these scenarios:

1. **Lossy-encoded streams**: Codecs like Opus and AAC alter the audio signal. Short clips (~1-2s) are especially affected because there is less signal to average out codec artifacts.
2. **Normal broadcast variation**: Even with lossless audio, radio shows have varying background music, room noise, and compression artifacts between broadcasts. A clean pattern is more robust against these day-to-day discrepancies.
3. **Repeating background sounds**: If the pattern clip contains repeating elements (e.g. background beeps, music loops, jingle tails) that also appear elsewhere in the audio stream, the detector may produce duplicate detections a few seconds apart from a single real occurrence. Filtering out the repeating frequency range eliminates these ghost matches.

## FFmpeg filter strategies

### Speech-range bandpass + denoise (recommended default)

```bash
ffmpeg -i input.wav -af "highpass=f=300,lowpass=f=3400,afftdn=nf=-25" -y output.wav
```

Removes frequencies outside the human speech range (300-3400 Hz) and applies FFT-based noise reduction. Works well for spoken jingles and vocal patterns. Also effective at removing repeating background beeps or tones that sit outside the speech range, which prevents duplicate detections from those sounds correlating with later occurrences in the stream.

### Narrow bandpass (for tonal patterns like bells/chimes)

```bash
ffmpeg -i input.wav -af "highpass=f=800,lowpass=f=2500,afftdn=nf=-20" -y output.wav
```

Aggressive filtering that isolates the dominant tone frequencies. Best for clips with distinct tonal content (bells, chimes, rolling sounds) where background instruments can be removed.

### Simple denoise (light touch)

```bash
ffmpeg -i input.wav -af "highpass=f=200,lowpass=f=3500,afftdn=nf=-25" -y output.wav
```

Gentler filtering that preserves more of the original signal. Good starting point when you're unsure of the frequency content.

## Synthesizing clean patterns

For tonal patterns (e.g. bell strikes, chimes), you can analyze the frequency content and synthesize a clean version with no background noise at all:

1. Analyze the clip's frequency content over time to identify dominant tones
2. Identify attack/decay envelope for each tone
3. Generate pure sine waves matching the detected frequencies and envelopes

This produces the highest Pearson r scores since the synthesized pattern has zero noise.

## Choosing a strategy

| Pattern type | Strategy | Expected improvement |
|---|---|---|
| Speech/jingles | Speech-range bandpass | Pearson r 0.6 -> 0.9 |
| Tonal (bells, chimes) | Narrow bandpass or synthesize | Pearson r 0.5 -> 0.93+ |
| Music with vocals | Speech-range bandpass | Moderate Pearson r gain |
| Short clips (<1.5s) | Narrow bandpass or synthesize | Most impactful |
| Long clips (>3s) | Light denoise or none | Minimal difference |
| Clips with repeating background sounds | Speech-range bandpass | Eliminates duplicate detections |

**General rule**: if a pattern clip contains sounds that repeat elsewhere in the broadcast (beeps, music loops, jingle tails), filter them out. The distinctive part of the clip — typically the speech or a unique tonal signature — is what the detector should match against. Leaving in repeating elements causes the detector to find those elements at multiple nearby offsets, producing cluster duplicates from a single real occurrence.

## afftdn parameter reference

- `nf`: Noise floor in dB. Range: -80 to -20. Lower = less aggressive. `-25` is a good default, `-20` is maximum aggression.
