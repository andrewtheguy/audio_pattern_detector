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
- Stdin modes (WAV, raw PCM, multiplexed)

For non-WAV file support (mp3, flac, etc.), pipe through ffmpeg on the host:

```shell
ffmpeg -i input.mp3 -f wav -ac 1 pipe: | \
  docker run --rm -i -v $(pwd)/pattern.wav:/pattern.wav:ro audio-pattern-detector \
  audio-pattern-detector match --stdin --pattern-file /pattern.wav
```

## Detection Algorithm

Currently only supports cross-correlation.

### Default Cross-Correlation

Picks all peaks that are above a certain threshold, and then eliminates false positives using partitioned MSE and multi-window Pearson correlation.

For normal patterns (>= 0.5s), the verification downsamples overlapping regions of the cross-correlation curves and computes the Pearson correlation coefficient across 3 windows (0-50%, 40-60%, 50-100%). This is scale-invariant, making it robust against lossy codec artifacts (Opus, AAC) that inflate the correlation envelope but preserve shape. High Pearson r (>= 0.90) can override moderate MSE, allowing detection even with degraded audio. See [docs/denoise-strategy.md](denoise-strategy.md) for improving pattern clip quality.

Works well with repeating or non-repeating patterns that are loud enough within the audio section because it adds the normalized clip at the end of the audio section, which helps to eliminate false positives that are much softer or non-related to the clip.

Short clips (< 0.5s) use the same verification but with a single 0-100% Pearson window and whole-only MSE (no middle partition emphasis), since the correlation envelope is too short for sub-region analysis. Short clips must cross-correlate well — this is the user's responsibility when providing the clip.

The RTHK hourly beep is a special case: it uses a dedicated pure tone verification path (triggered by the clip's `strategy` field, declared in a `.apd` pattern config) because a clean pure tone does not cross-correlate well enough for the normal path. This is independent of the short clip path. See [docs/pattern-matching.md](pattern-matching.md) for the `.apd` format.

It will miss distorted patterns like this because error score is too high and Pearson r is too low:

![rthk_beep_39_00:39:00_478782](https://github.com/user-attachments/assets/80669708-b8f9-461c-ae6c-2edddb161904)
