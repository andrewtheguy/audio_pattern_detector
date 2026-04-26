# Pattern Matching Flow

This document describes how audio pattern detection works, from raw audio data to final match results.

## Overview

Detection is a **two-step process**:

- **Step 1 — Candidate detection.** FFT-based cross-correlation against the audio always runs first. Its job is to find and center potential match locations as candidate peaks. Every clip type goes through this step, regardless of length or strategy.
- **Step 2 — Candidate verification.** Each candidate peak from Step 1 is then verified using one of three paths, chosen by clip type: normal verification (for regular clips ≥ 0.5s), short-clip verification (for clips < 0.5s), or marker-tone verification (for `.apd.toml` patterns with `strategy = "marker_tone"`). The three paths are alternatives at Step 2 — they all share the same Step 1.

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
─── Step 1: Candidate detection (always runs) ───────────────────
        |
   FFT cross-correlation against each clip
        |
   Peak detection (height >= 0.25, min distance = clip length)
        |
─── Step 2: Candidate verification (one path per candidate) ─────
        |
   For each candidate peak, dispatch on clip type:
        |
   ┌─ strategy == "marker_tone"? ─────────────────────────────┐
   │  YES → Marker-tone verification                           │
   │        (narrowband spectral check at dominant frequency)  │
   │                                                           │
   │  NO  → Correlation-envelope verification:                 │
   │        • clip >= 0.5s  → Normal: partitioned MSE          │
   │                          + 3-window Pearson r             │
   │        • clip <  0.5s  → Short clip: whole-only MSE       │
   │                          + single-window Pearson r        │
   └───────────────────────────────────────────────────────────┘
        |
   Accepted peaks converted to timestamps
```

## Pre-computation (Initialization)

Before processing audio, each clip is prepared once:

1. **Loudness normalization** - clip audio normalized to -16 dB LUFS.
2. **Self-correlation** - FFT cross-correlation of the clip with itself (`fft_correlate_1d(clip, clip, mode='full')`), producing a reference correlation curve. The absolute max is stored for normalization later.
3. **Tone-strategy setup** - for clips whose strategy is `marker_tone`, the dominant frequency is recorded as clip metadata. All other clips set `dominant_frequency = None`.

This produces a `ClipData` dict per clip containing the normalized audio, clip name, sliding window, self-correlation curve, its absolute max, and the dominant frequency (non-None only for tone-strategy clips).

## Chunked Processing

Audio is read as a stream of float32 samples and split into fixed-size chunks (`seconds_per_chunk`, default 60s). Here, `chunk` means the current chunk being processed and `previous_chunk` means the immediately preceding chunk, if one exists.

The raw chunks themselves are not read with overlap. Instead, for each clip, the detector builds an `audio_section`. In the normal case, it prepends the last `sliding_window` seconds from `previous_chunk` to `chunk`, where `sliding_window = ceil(clip_duration_seconds)`. This ensures patterns near chunk boundaries are not missed.

For the final chunk, if `len(chunk) / sample_rate < seconds_per_chunk`, the detector does not use the usual `sliding_window` prepend. Instead, it first concatenates `previous_chunk` and `chunk`, then extracts the last `seconds_per_chunk` seconds from that combined buffer to form `audio_section`.

Each per-clip `audio_section` is loudness-normalized independently to -16 dB LUFS before correlation.

## Step 1: Candidate Detection (FFT Cross-Correlation)

This step always runs first for every clip type. Its job is to locate and center potential match positions; it does not decide whether a candidate is a true match — that is Step 2.

For each chunk and each clip:

1. Compute `fft_correlate_1d(audio_section, clip, mode='full')` and take the absolute value.
2. Normalize by `max(self_correlation_max, cross_correlation_max)` so the correlation curve is in [0, 1].
3. Run peak detection with `height >= 0.25` and `distance >= clip_length` (prevents duplicate detections within one clip duration).

Each peak is a candidate match location. Candidates are discarded only if the centered slice would extend more than 5 samples beyond the correlation array; otherwise zero-padding is used to keep the slice length consistent.

## Step 2: Candidate Verification

Every candidate peak from Step 1 is verified before being accepted as a match. The verifier branch is chosen by clip type, and the three branches below are alternatives — exactly one runs per candidate:

- **Normal Patterns** (`.wav` clips ≥ 0.5s) — partitioned MSE plus 3-window Pearson r on the centered correlation slice.
- **Short Clips** (`.wav` clips < 0.5s) — same correlation-envelope approach but with simplified single-window MSE and Pearson.
- **Marker Tone** (`.apd.toml` clips with `strategy = "marker_tone"`) — narrowband spectral check at the declared dominant frequency, instead of the correlation-envelope shape check.

For the two correlation-envelope paths (Normal and Short Clips), a slice of the cross-correlation curve centered on the peak is extracted, with the same length as the self-correlation curve. Zero-padding is applied if needed at the ends, and the slice is normalized by its own max. The self-correlation curve acts as the "ideal" shape; the verification asks: does this candidate's correlation slice look like the ideal? The marker-tone path skips this shape comparison entirely and works on the candidate's audio segment directly.

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

Short clips go through the normal correlation-envelope path but with simplified windowing:

1. **MSE** — only `similarity_whole` is used (no middle partition emphasis). The correlation envelope is too short for sub-region analysis to be meaningful.

2. **Pearson correlation** — a single 0-100% window (505 downsampled points) instead of the three partial windows. The full window captures the overall shape without splitting into regions that would be too small.

3. **Same thresholds** — `similarity > 0.03` rejects, `pearson_r >= 0.90` accepts.

**Limitation**: short clips require good cross-correlation characteristics. The clip must produce a distinctive correlation peak that matches its self-correlation envelope. Clips that don't cross-correlate well (e.g. clean pure tones like the RTHK hourly beep) need their own special-case path instead — see `.apd.toml` pattern configs below.

### Marker Tone (`.apd.toml` clips)

Marker-tone clips still go through Step 1 — FFT cross-correlation against the synthesised clip and peak detection — to find and center candidate locations. Only Step 2 differs: instead of the correlation-envelope shape check used by Normal Patterns and Short Clips, the candidate's audio segment is verified with a narrowband spectral check at the clip's declared dominant frequency. This is needed because clean pure tones (like the RTHK hourly beep) do not produce a distinctive enough correlation envelope for the shape-based path to verify reliably.

Patterns that need this special detection strategy use a `.apd.toml` file instead of `.wav`. The file is a plain TOML document (parsed via `tomllib` in the stdlib, with `#` comments) that declares the strategy plus a generator used to synthesise the pattern clip at the target sample rate. Ordinary patterns continue to use `.wav`.

Example (`sample_audios/clips/rthk_beep.apd.toml`):

```toml
strategy = "marker_tone"
description = "RTHK hourly beep — ~1040 Hz pure tone, ~0.23s"

[generator]
type = "sine"
frequency_hz = 1040.19
duration_seconds = 0.228375
amplitude = 1.0
```

The currently implemented tone strategy is `marker_tone`; the only generator is `sine`. The extension point is the `strategy` field — adding a new special handling means adding a new strategy name and wiring it in `audio_pattern_detector.py` and `pattern_config.py`.

When a clip's strategy is tone-based:
1. The dominant frequency declared in `generator.frequency_hz` is stored as the clip's strategy parameter during initialisation (no FFT re-derivation is needed, though `get_pure_tone_frequency()` is used as a fallback if absent).
2. At verification time, `_verify_marker_tone` checks the candidate audio segment for narrowband energy at the expected frequency using short-time spectral analysis.
3. Per-clip `verification` parameters in the TOML file tune how much in-window purity and adjacent-flank leakage are allowed for that station's marker.

Because the clip is synthesised at the target sample rate, a single `.apd.toml` file works at 8 kHz, 16 kHz, or any other supported rate without needing a per-rate variant.

## Pure Tone Classification

A clip is classified as a pure tone if its frequency spectrum (via FFT) has exactly one prominent peak (prominence > 0.05 in the normalized magnitude spectrum) matching the dominant frequency within 1% relative tolerance. This classification backs the tone-based `.apd.toml` strategies.

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
