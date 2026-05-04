# Development

## Type Checking

Use basedpyright for static type checking:

```shell
uv run basedpyright
```

## Linting

```shell
uv run ruff check
```

## Testing

Use pytest to test because not all of them are written using default python unittest module, and pytest is more flexible and easier to use.

```shell
uv run pytest
```

## Native Helper Rebuild

After changing Rust code in `native-helper/src/`, rebuild the Python extension:

```shell
uv run maturin develop --skip-install --manifest-path native-helper/Cargo.toml
```

## Debug graphs

matplotlib is required for `--debug` graph output but is not installed by default. Enable it with:

```shell
uv sync --group debug
```

This installs the `debug` dependency group (which includes `dev` plus matplotlib). Only needed during local development when tuning detection parameters.

## Docker

### Testing with Docker

Use `Dockerfile.test` which includes ffmpeg and dev dependencies:

```shell
# Build test image
docker build -f Dockerfile.test -t audio-pattern-detector-test .

# Run tests (mount tests and sample_audios since they're excluded from image)
docker run --rm \
  -v $(pwd)/tests:/usr/src/app/tests:ro \
  -v $(pwd)/sample_audios:/usr/src/app/sample_audios:ro \
  audio-pattern-detector-test
```

### Production Docker build

Use the default `Dockerfile` which is minimal and does not include ffmpeg:

```shell
docker build -t audio-pattern-detector .
```

**Note**: The production image does not include ffmpeg. It supports:
- WAV files (processed with scipy, no ffmpeg needed)
- Stdin modes (WAV, multiplexed)

For non-WAV file support (mp3, flac, etc.), pipe through ffmpeg on the host:

```shell
ffmpeg -i input.mp3 -f wav -ac 1 pipe: | \
  docker run --rm -i -v $(pwd)/pattern.wav:/pattern.wav:ro audio-pattern-detector \
  audio-pattern-detector match --stdin --pattern-file /pattern.wav
```

## Detection Algorithm

Detection is a **two-step process**:

- **Step 1 — Candidate detection**: FFT cross-correlation against the audio section, followed by peak detection (height ≥ 0.25, min distance = clip length). This step always runs first for every clip type and produces centered candidate match locations.
- **Step 2 — Candidate verification**: each candidate from Step 1 is verified by exactly one of three branches, chosen by clip type. The branches below are alternatives — they all share Step 1.

### Step 2 paths

**Normal Patterns (`.wav` clips ≥ 0.5s)** — verification uses partitioned MSE plus multi-window Pearson correlation. The cross-correlation curve is downsampled across 3 overlapping regions (0-50%, 40-60%, 50-100%) and compared against the clip's self-correlation. Pearson r is scale-invariant, making it robust against lossy codec artifacts (Opus, AAC) that inflate the correlation envelope but preserve shape. High Pearson r (≥ 0.90) can override moderate MSE, allowing detection even with degraded audio. See [docs/denoise-strategy.md](denoise-strategy.md) for improving pattern clip quality. Works well with repeating or non-repeating patterns that are loud enough within the audio section because the normalized clip is appended to the end of the audio section, which helps eliminate false positives that are much softer or unrelated.

**Short Clips (`.wav` clips < 0.5s)** — uses the same correlation-envelope approach as Normal Patterns but with simplified windowing: a single 0-100% Pearson window and whole-only MSE (no middle partition emphasis), since the correlation envelope is too short for sub-region analysis. Short clips must cross-correlate well at Step 1 — this is the user's responsibility when providing the clip.

**Marker Tone (`.apd.toml` clips with `strategy = "marker_tone"`)** — used for things like the RTHK hourly station beep, where a clean sine does not produce a distinctive enough Step-1 envelope for the shape-based paths to verify reliably. Step 1 still runs (cross-correlation against the synthesised tone clip + peak detection); Step 2 substitutes a narrowband spectral check at the declared dominant frequency. Each `.apd.toml` file can carry its own `verification` thresholds, so stations like RTHK, 881, and 903 can tune the same verifier without separate strategy code. This is independent of the short clip path. See [docs/pattern-matching.md](pattern-matching.md) for the `.apd.toml` format.

The Normal/Short paths will miss distorted patterns like this because error score is too high and Pearson r is too low:

![rthk_beep_39_00:39:00_478782](https://github.com/user-attachments/assets/80669708-b8f9-461c-ae6c-2edddb161904)
