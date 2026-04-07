# Pattern Matching Flow

This document describes how audio pattern detection works, from raw audio data to final match results.

## Overview

The detector finds occurrences of short audio patterns (clips) within a longer audio stream using FFT-based cross-correlation, followed by similarity verification to reject false positives. There are two verification paths based on clip duration: one for short clips (< 0.5s) using a single-window Pearson r check, and one for longer clips using a multi-window Pearson r check.

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
        |--- Pure tone pattern ---> Downsampled MSE + Pearson r check (0-100%)
        |--- Normal pattern -----> Partitioned MSE + multi-window Pearson r check
        |
   Accepted peaks converted to timestamps
```

## Pre-computation (Initialization)

Before processing audio, each clip is prepared once:

1. **Loudness normalization** - clip audio normalized to -16 dB LUFS.
2. **Self-correlation** - FFT cross-correlation of the clip with itself (`fft_correlate_1d(clip, clip, mode='full')`), producing a reference correlation curve. The absolute max is stored for normalization later.
3. **Short clip classification** - clips shorter than 0.5s are classified as short clips and use a simplified single-window Pearson r verification path.

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

### Short Clips (< 0.5s)

Short clips have fewer samples in their correlation curves, making multi-window analysis unreliable. The verification uses the same Pearson correlation approach as normal patterns, but with a single window covering the full 0-100% range. Both curves are downsampled to 101 points before comparison, preserving local maxima.

1. **Downsampled MSE** - MSE between the two 101-point curves.
2. **Pearson correlation** - Pearson r is computed on the same 101-point downsampled curves covering 0-100% of the correlation envelope. A single full-range window suffices because short clip correlation envelopes don't benefit from partial-window analysis.

3. **Decision thresholds**:
   - `similarity > 0.01` -> reject (hard MSE ceiling, tighter than normal patterns)
   - `pearson_r >= 0.90` -> accept
   - Otherwise -> reject

## Short Clip Classification

A clip is classified as a short clip if its duration is less than 0.5 seconds. Short clips use a single-window Pearson r verification path instead of the multi-window approach used for longer clips.

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
| `ClipCache` | Runtime cache: downsampled Pearson windows per clip |
