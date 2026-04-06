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

Picks all peaks that are above a certain threshold, and then eliminates false positives using partitioned MSE and Pearson correlation (for normal patterns) or downsampled MSE and overlap ratio (for pure tone patterns).

For normal patterns, the verification downsamples the middle region of the cross-correlation curves to 101 points, then computes the Pearson correlation coefficient. This is scale-invariant, making it robust against lossy codec artifacts (Opus, AAC) that inflate the correlation envelope but preserve shape. High Pearson r (>= 0.85) can override moderate MSE, allowing detection even with degraded audio. See [docs/denoise-strategy.md](denoise-strategy.md) for improving pattern clip quality.

Works well with repeating or non-repeating patterns that are loud enough within the audio section because it adds the normalized clip at the end of the audio section, which helps to eliminate false positives that are much softer or non-related to the clip.

Won't work well for patterns that are too short. Currently it disallows short clips unless it is a pure tone pattern. If it is short and a pure tone pattern, then a special correlation logic is used to match.

It will miss distorted patterns like this because error score is too high and Pearson r is too low:

![rthk_beep_39_00:39:00_478782](https://github.com/user-attachments/assets/80669708-b8f9-461c-ae6c-2edddb161904)
