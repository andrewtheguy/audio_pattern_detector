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

## Testing with Docker

Build with dev dependencies and run tests with mounted directories:

```shell
# Build with dev dependencies
docker build --build-arg INCLUDE_DEV=true -t audio-pattern-detector-test .

# Run tests (mount tests and sample_audios since they're excluded from image)
docker run --rm \
  -v $(pwd)/tests:/usr/src/app/tests:ro \
  -v $(pwd)/sample_audios:/usr/src/app/sample_audios:ro \
  audio-pattern-detector-test \
  uv run pytest -v
```

## Production Docker build

```shell
# Default build (no dev dependencies)
docker build -t audio-pattern-detector .
```

## Detection Algorithm

Currently only supports cross-correlation.

### Default Cross-Correlation

Picks all peaks that are above a certain threshold, and then eliminates false positives with cross similarity and mean square error.

Works well with repeating or non-repeating patterns that are loud enough within the audio section because it adds the normalized clip at the end of the audio section, which helps to eliminate false positives that are much softer or non-related to the clip.

Won't work well for patterns that are too short. Currently it disallows short clips unless it is a pure tone pattern. If it is short and a pure tone pattern, then a special correlation logic is used to match.

It will miss distorted patterns like this because error score is too high and area overlap ratio is too low:

![rthk_beep_39_00:39:00_478782](https://github.com/user-attachments/assets/80669708-b8f9-461c-ae6c-2edddb161904)
