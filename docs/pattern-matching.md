# Pattern Matching Flow

This document describes how audio pattern detection works, from raw audio data to final match results.

## Overview

The detector finds occurrences of short audio patterns (clips) within a longer audio stream using FFT-based cross-correlation, followed by similarity verification to reject false positives. There are two verification paths: one for normal patterns and one for pure tone patterns (e.g. beeps).

## Pipeline

```
Raw audio stream (float32 PCM)
        |
   Read fixed-size chunks (default 60s)
        |
   Build a per-clip audio section with previous-chunk context
        |
   Loudness-normalize the audio section to -16 dB LUFS
        |
   FFT cross-correlation against each clip
        |
   Peak detection (height >= 0.25, min distance = clip length)
        |
   For each candidate peak:
        |--- Pure tone pattern ---> Downsampled MSE + overlap ratio check
        |--- Normal pattern -----> Partitioned MSE + multi-window Pearson r check
        |
   Accepted peaks converted to timestamps
```

## Pre-computation (Initialization)

Before processing audio, each clip is prepared once:

1. **Loudness normalization** - clip audio normalized to -16 dB LUFS.
2. **Self-correlation** - FFT cross-correlation of the clip with itself (`fft_correlate_1d(clip, clip, mode='full')`), producing a reference correlation curve. The absolute max is stored for normalization later.
3. **Pure tone detection** - FFT of the clip is analyzed: if the spectrum has exactly one prominent peak matching the dominant frequency, the clip is classified as a pure tone.

This produces a `ClipData` dict per clip containing the normalized audio, clip name, sliding window, self-correlation curve, and its absolute max. Pure-tone classification is pre-computed separately in `ClipCache`.

## Chunked Processing

Audio is read as a stream of float32 samples and split into fixed-size chunks (`seconds_per_chunk`, default 60s). Here, `chunk` means the current chunk being processed and `previous_chunk` means the immediately preceding chunk, if one exists.

The raw chunks themselves are not read with overlap. Instead, for each clip, the detector builds an `audio_section`. In the normal case, it prepends the last `sliding_window` seconds from `previous_chunk` to `chunk`, where `sliding_window = ceil(clip_duration_seconds)`. This ensures patterns near chunk boundaries are not missed.

For the final chunk, if `len(chunk) / sample_rate < seconds_per_chunk`, the detector does not use the usual `sliding_window` prepend. Instead, it first concatenates `previous_chunk` and `chunk`, then extracts the last `seconds_per_chunk` seconds from that combined buffer to form `audio_section`.

Each per-clip `audio_section` is loudness-normalized independently to -16 dB LUFS before correlation.

## FFT Cross-Correlation

For each chunk and each clip:

1. Compute `fft_correlate_1d(audio_section, clip, mode='full')` and take the absolute value.
2. Normalize by `max(self_correlation_max, cross_correlation_max)` so the correlation curve is in [0, 1].
3. Run peak detection with `height >= 0.25` and `distance >= clip_length` (prevents duplicate detections within one clip duration).

Each peak is a candidate match location. Candidates are discarded only if the centered slice would extend more than 5 samples beyond the correlation array; otherwise zero-padding is used to keep the slice length consistent.

## Similarity Verification

For each candidate peak, a slice of the cross-correlation curve centered on the peak is extracted, with the same length as the self-correlation curve. Zero-padding is applied if needed at the ends, and the slice is normalized by its own max.

The self-correlation curve acts as the "ideal" shape. The verification asks: does this candidate's correlation slice look like the ideal?

### Normal Patterns

Verification uses partitioned mean squared error (MSE) plus Pearson correlation of the downsampled envelope:

1. **Partitioned MSE** - both curves are divided into 10 equal partitions. MSE is computed per partition. Two summary metrics are derived:
   - `similarity_middle`: mean MSE of partitions 4-5 (the center 20%)
   - `similarity_whole`: mean MSE across all 10 partitions
   - Final `similarity = min(similarity_whole, similarity_middle)`

   The middle partitions are checked separately because real distortions tend to appear there.

2. **Multi-window Pearson correlation** - three overlapping regions of the curves are compared to find the best shape match:
   - Window A: first half (partitions 0-4, 0-50%), downsampled to 252 points
   - Window B: center (partitions 4-5, 40-60%), downsampled to 101 points
   - Window C: second half (partitions 5-9, 50-100%), downsampled to 252 points

   Each window is downsampled using `resample_preserve_maxima` (sample count proportional to window width for consistent resolution), then the Pearson correlation coefficient is computed between the pattern's self-correlation window and the candidate's cross-correlation window. The best (highest) Pearson r across the three windows is used.

   Pearson r is scale-invariant — it measures shape similarity regardless of amplitude differences. This is important for lossy-encoded audio (e.g. Opus HLS streams) where codec artifacts inflate the correlation envelope but preserve the overall shape. The multi-window approach handles cases where the peak shape is slightly asymmetric or off-center.

3. **Decision thresholds**:
   - `similarity > 0.03` -> reject (hard MSE ceiling)
   - `pearson_r >= 0.90` -> accept (shape matches well, even if MSE is moderately elevated)
   - Otherwise -> reject (shape doesn't match well enough)

### Pattern Clip Quality

Detection accuracy depends heavily on the quality of the pattern clip. Clips with background noise, repeating sounds (beeps, music loops), or frequencies outside the distinctive signal range can cause:
- **False negatives**: noise widens the cross-correlation envelope, reducing Pearson r below the threshold
- **Duplicate detections**: repeating elements in the clip (e.g. background beeps) correlate with later occurrences of those same sounds in the stream, producing ghost matches a few seconds after the real detection

Denoising the pattern clip with bandpass filtering (e.g. speech range 300-3400 Hz) removes these issues. For tonal patterns, synthesizing a clean version from the dominant frequencies produces the best results. See [denoise-strategy.md](denoise-strategy.md).

### Pure Tone Patterns

Pure tones (beeps) have repetitive, high-frequency correlation curves that are sensitive to slight shifts. To handle this, both curves are downsampled to 101 points before comparison, preserving local maxima in each window.

1. **Downsampled MSE** - MSE between the two 101-point curves.
2. **Overlap ratio** - compute Simpson integrals on the two downsampled curves:
   - `area_control`: area under the downsampled reference/self-correlation curve
   - `area_y2`: area under the downsampled candidate correlation slice
   - `overlapping_area`: area under the pointwise minimum of those two curves

   The metric used by the code is `overlap_ratio = overlapping_area / area_control`, so the denominator is the reference curve's area, not the candidate curve's area or a rectangular normalization term.

3. **Decision thresholds**:
   - `similarity > 0.01` -> reject
   - `0.003 < similarity <= 0.01` and `overlap_ratio < 0.99` -> reject
   - `0.002 < similarity <= 0.003` and `overlap_ratio < 0.98` -> reject
   - `similarity <= 0.002` -> accept

Pure tone thresholds require higher overlap ratios because a matching beep should almost perfectly overlap the reference.

## Pure Tone Classification

A clip is classified as a pure tone if its frequency spectrum (via FFT) has exactly one prominent peak (prominence > 0.05 in the normalized magnitude spectrum) and that peak's frequency matches the dominant frequency within 1% relative tolerance.

## Timestamp Conversion

Accepted peaks (in sample indices) are converted to timestamps in fractional seconds (`float`):

1. Subtract the section offset used to build `audio_section`:
   - `0` for the first chunk
   - usually `sliding_window` for later full chunks
   - the negated "missing time" value for the final short chunk. Here, "missing time" means `actual_chunk_duration - seconds_per_chunk`, so it is negative when the final `chunk` is shorter than expected; for example, if `seconds_per_chunk = 10s` but the final `chunk` is only `6s`, the missing time is `6 - 10 = -4s`, and the code subtracts `-(-4s) = 4s` because it had to borrow `4s` from `previous_chunk` when building the final `audio_section`.
2. Add the chunk's offset from the start of the stream (`index * seconds_per_chunk`).
3. Shift backward by the clip duration so the timestamp marks the start of the pattern rather than the correlation peak.
4. Clamp negative results to `0`.

## Key Data Structures

| Structure | Description |
|-----------|-------------|
| `AudioClip` | Input pattern: name, audio array, sample rate |
| `ClipData` | Pre-computed per clip: normalized audio, clip name, self-correlation curve, absolute max, sliding window |
| `ClipCache` | Runtime cache: pure-tone classification flags and downsampled self-correlation curves for pure-tone clips |
| `OverlapResult` | Return type from `area_of_overlap_ratio` (used by pure tone path). The control areas are the baseline areas for the reference/control curve: `area_control` is the Simpson area of that reference region, and `total_rect_control` is its bounding-rectangle area. `area_y2` is the candidate curve area. These fields feed the derived ratios, including `overlap_ratio = overlapping_area / area_control`. |
