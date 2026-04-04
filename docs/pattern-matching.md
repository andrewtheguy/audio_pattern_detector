# Pattern Matching Flow

This document describes how audio pattern detection works, from raw audio data to final match results.

## Overview

The detector finds occurrences of short audio patterns (clips) within a longer audio stream using FFT-based cross-correlation, followed by similarity verification to reject false positives. There are two verification paths: one for normal patterns and one for pure tone patterns (e.g. beeps).

## Pipeline

```
Raw audio stream (float32 PCM)
        |
   Chunk into segments (default 60s, with overlap)
        |
   Loudness-normalize chunk to -16 dB LUFS
        |
   FFT cross-correlation against each clip
        |
   Peak detection (height > 0.25, min distance = clip length)
        |
   For each candidate peak:
        |--- Pure tone pattern ---> Downsampled MSE + overlap ratio check
        |--- Normal pattern -----> Partitioned MSE + area overlap check
        |
   Accepted peaks converted to timestamps
```

## Pre-computation (Initialization)

Before processing audio, each clip is prepared once:

1. **Loudness normalization** - clip audio normalized to -16 dB LUFS.
2. **Self-correlation** - FFT cross-correlation of the clip with itself (`fft_correlate_1d(clip, clip, mode='full')`), producing a reference correlation curve. The absolute max is stored for normalization later.
3. **Pure tone detection** - FFT of the clip is analyzed: if the spectrum has exactly one prominent peak matching the dominant frequency, the clip is classified as a pure tone.

This produces a `ClipData` dict per clip containing the normalized audio, self-correlation curve, and its absolute max.

## Chunked Processing

Audio is read as a stream of float32 samples and split into fixed-size chunks (`seconds_per_chunk`, default 60s).

Each chunk overlaps with the previous one by `sliding_window` seconds (ceil of the clip duration). This ensures patterns near chunk boundaries are not missed. For the last chunk (which may be shorter than `seconds_per_chunk`), the overlap uses the full remaining length.

Each chunk is loudness-normalized independently to -16 dB LUFS before correlation.

## FFT Cross-Correlation

For each chunk and each clip:

1. Compute `fft_correlate_1d(audio_section, clip, mode='full')` and take the absolute value.
2. Normalize by `max(self_correlation_max, cross_correlation_max)` so the correlation curve is in [0, 1].
3. Run peak detection with `height >= 0.25` and `distance >= clip_length` (prevents duplicate detections within one clip duration).

Each peak is a candidate match location. Candidates near the boundaries (within 5 samples of out-of-bounds) are discarded.

## Similarity Verification

For each candidate peak, a slice of the cross-correlation curve centered on the peak is extracted, with the same length as the self-correlation curve. This slice is normalized by its own max.

The self-correlation curve acts as the "ideal" shape. The verification asks: does this candidate's correlation slice look like the ideal?

### Normal Patterns

Verification uses partitioned mean squared error (MSE) plus area overlap:

1. **Partitioned MSE** - both curves are divided into 10 equal partitions. MSE is computed per partition. Two summary metrics are derived:
   - `similarity_middle`: mean MSE of partitions 4-5 (the center 20%)
   - `similarity_whole`: mean MSE across all 10 partitions
   - Final `similarity = min(similarity_whole, similarity_middle)`

   The middle partitions are checked separately because real distortions tend to appear there.

2. **Area overlap ratio** - for the middle 40% of the curves (partitions 4-6), compute the area of each curve via Simpson's rule, the overlapping area (integral of the pointwise minimum), and the non-overlapping area. The metric is `diff_overlap_ratio = non_overlapping_area / overlapping_area`.

3. **Decision thresholds**:
   - `similarity > 0.01` -> reject
   - `0.002 < similarity <= 0.01` and `diff_overlap_ratio > 0.5` -> reject
   - `similarity <= 0.002` -> accept (no area check needed)

### Pure Tone Patterns

Pure tones (beeps) have repetitive, high-frequency correlation curves that are sensitive to slight shifts. To handle this, both curves are downsampled to 101 points before comparison, preserving local maxima in each window.

1. **Downsampled MSE** - MSE between the two 101-point curves.
2. **Overlap ratio** - `overlapping_area / area_control` via Simpson's rule on the downsampled curves.

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

1. Subtract the `sliding_window` overlap offset.
2. Add the chunk's offset from the start of the stream (`index * seconds_per_chunk`).
3. Shift backward by the clip duration so the timestamp marks the start of the pattern rather than the correlation peak.

## Key Data Structures

| Structure | Description |
|-----------|-------------|
| `AudioClip` | Input pattern: name, audio array, sample rate |
| `ClipData` | Pre-computed per clip: normalized audio, self-correlation curve, absolute max, sliding window |
| `ClipCache` | Runtime cache: downsampled correlation clips (for pure tones), pure tone classification flags |
| `OverlapResult` | Area metrics from overlap calculation: overlapping area, diff area, control area, ratios |
