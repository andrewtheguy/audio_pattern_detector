# Audio Pattern Detector

Detects audio patterns specified by audio clips in target audio files. Designed to detect intros, breaks, and outros from prerecorded radio shows and podcasts.

Useful for AI workflows to efficiently segment audio files before processing (e.g., OpenAI Whisper transcription preprocessing).

Uses cross-correlation to detect potential matches, then uses mean square error and Pearson correlation to eliminate false positives. Robust against lossy-encoded audio (Opus, AAC).

## Installation

### Install from GitHub Pages package index (recommended)

Automatically selects the correct wheel for your platform (Linux x86_64, Linux arm64, macOS Apple Silicon).

```shell
uv tool install \
  --extra-index-url https://andrewtheguy.github.io/audio_pattern_detector/simple/ \
  --extra-index-url https://andrewtheguy.github.io/fft-correlation/simple/ \
  --extra-index-url https://andrewtheguy.github.io/andrew_utils/simple/ \
  'audio-pattern-detector==x.x.x'
```

The extra `fft-correlation` and `andrew-utils` indexes are required because those transitive dependencies are not published to PyPI.

### Install from source (requires Rust toolchain)
```shell
uv tool install git+https://github.com/andrewtheguy/audio_pattern_detector.git@(tag or branch)
```

### Run without installing
```shell
uv tool run --from git+https://github.com/andrewtheguy/audio_pattern_detector.git@vx.x.x audio-pattern-detector [command] [options]

# Or from local directory
uv run audio-pattern-detector [command] [options]
```

## Audio Requirements

- **Mono Only**: Only mono (single channel) audio is supported
- **Sample Rate**: Default is 8kHz (configurable via `--target-sample-rate`)
- **Format**: WAV files recommended (no ffmpeg required). Non-WAV files need ffmpeg.

## Quick Start

### Match - Detect patterns in audio

```shell
# Basic usage
audio-pattern-detector match audio.wav --pattern-file pattern.wav

# With multiple patterns
audio-pattern-detector match audio.wav --pattern-folder ./patterns/

# Streaming from stdin
ffmpeg -i input.mp3 -f wav -ac 1 -ar 8000 pipe: | \
  audio-pattern-detector match --stdin --pattern-file pattern.wav
```

### Show-config - Show computed configuration

```shell
audio-pattern-detector show-config ./clips/rthk_beep.apd.toml
```

## CLI Options (match)

| Option                 | Description                                                              |
|------------------------|--------------------------------------------------------------------------|
| `audio_file`           | Audio file to search for patterns (positional argument)                  |
| `--stdin`              | Read WAV audio from stdin                                                |
| `--multiplexed-stdin`  | Read patterns and audio from stdin via binary protocol (for IPC)         |
| `--target-sample-rate` | Target sample rate for processing (default: 8000)                        |
| `--pattern-file`       | Single pattern file (WAV)                                                |
| `--pattern-folder`     | Folder of pattern clips (WAV)                                            |
| `--chunk-seconds`      | Seconds per chunk (default: 60, or "auto")                               |
| `--timestamp-format`   | JSONL timestamp fields: `both` (default), `ms`, or `formatted`           |
| `--debug`              | Enable debug mode                                                        |
| `--debug-dir`          | Base directory for debug output (default: ./tmp)                         |

## JSONL Output Format

Output is always streaming JSONL:

By default, JSONL timestamp events include both millisecond and formatted
fields. Use `--timestamp-format ms` or `--timestamp-format formatted` to emit
just one representation.

```jsonl
{"type": "start", "source": "audio.wav"}
{"type": "pattern_detected", "clip_name": "pattern", "timestamp_ms": 5500, "timestamp_formatted": "00:00:05.500"}
{"type": "end", "total_time_ms": 60000, "total_time_formatted": "00:01:00.000"}
```

## Documentation

- **[Pattern Matching](docs/pattern-matching.md)** - Detailed description of the detection pipeline, verification logic, and thresholds
- **[Denoise Strategy](docs/denoise-strategy.md)** - How to denoise pattern clips for better matching with lossy-encoded or noisy audio
- **[Stdin Modes](docs/stdin-modes.md)** - WAV stdin, raw PCM stdin, and multiplexed stdin (IPC) with code examples for Node.js, Python, Go
- **[Development](docs/development.md)** - Type checking, linting, testing, Docker, detection algorithm details

## Development

```shell
uv run basedpyright  # Type checking
uv run ruff check    # Linting
uv run pytest        # Testing

# Rebuild native-helper after changing Rust files in native-helper/src/
uv run maturin develop --skip-install --manifest-path native-helper/Cargo.toml
```

See [docs/development.md](docs/development.md) for more details.
