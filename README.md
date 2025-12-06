# Audio Pattern Detector

Detects audio patterns specified by audio clips in target audio files. Designed to detect intros, breaks, and outros from prerecorded radio shows and podcasts.

Useful for AI workflows to efficiently segment audio files before processing (e.g., OpenAI Whisper transcription preprocessing).

Uses cross-correlation to detect potential matches, then uses mean square error and overlapping areas to eliminate false positives.

## Installation

### Install from GitHub (recommended)
```shell
uv tool install git+https://github.com/andrewtheguy/audio_pattern_detector.git@ref(tag or branch)
```

### Install locally for development
```shell
uv pip install -e .
```

### Install with optional debug dependencies
```shell
uv pip install -e ".[debug]"
```

This installs matplotlib for debug visualizations.

### Run without installing
```shell
# Using uv (from local directory)
uv run audio-pattern-detector [command] [options]

# Using pipx (from GitHub)
pipx run --spec git+https://github.com/andrewtheguy/audio_pattern_detector.git audio-pattern-detector [command] [options]
```

## Audio Requirements

- **Mono Only**: Only mono (single channel) audio is supported
- **Sample Rate**: Default is 8kHz (configurable via `--target-sample-rate`)
- **Format**: WAV files recommended (no ffmpeg required). Non-WAV files need ffmpeg.

## Quick Start

### Match - Detect patterns in audio

```shell
# Basic usage
audio-pattern-detector match --audio-file audio.wav --pattern-file pattern.wav

# With multiple patterns
audio-pattern-detector match --audio-file audio.wav --pattern-folder ./patterns/

# Streaming from stdin (outputs JSONL)
ffmpeg -i input.mp3 -f wav -ac 1 pipe: | \
  audio-pattern-detector match --stdin --pattern-file pattern.wav
```

### Convert - Convert audio to clip format

```shell
audio-pattern-detector convert --audio-file input.wav --dest-file output.wav
```

### Show-config - Show computed configuration

```shell
audio-pattern-detector show-config --pattern-folder ./clips
```

## CLI Options (match)

| Option                 | Description                                                              |
|------------------------|--------------------------------------------------------------------------|
| `--audio-file`         | Audio file to search for patterns                                        |
| `--audio-folder`       | Folder of audio files to process                                         |
| `--stdin`              | Read WAV audio from stdin (outputs JSONL)                                |
| `--multiplexed-stdin`  | Read patterns and audio from stdin via binary protocol (for IPC)         |
| `--raw-pcm`            | Stdin is raw float32 PCM instead of WAV                                  |
| `--source-sample-rate` | Source sample rate for raw PCM (required with `--raw-pcm`)               |
| `--target-sample-rate` | Target sample rate for processing (default: 8000)                        |
| `--pattern-file`       | Single pattern file (WAV)                                                |
| `--pattern-folder`     | Folder of pattern clips (WAV)                                            |
| `--chunk-seconds`      | Seconds per chunk (default: 60, or "auto")                               |
| `--jsonl`              | Output JSONL events for file inputs                                      |
| `--debug`              | Enable debug mode                                                        |

## JSONL Output Format

When using `--stdin`, `--multiplexed-stdin`, or `--jsonl`:

```jsonl
{"type": "start", "source": "audio.wav"}
{"type": "pattern_detected", "clip_name": "pattern", "timestamp": 5.5, "timestamp_formatted": "00:00:05.500"}
{"type": "end", "total_time": 60.0, "total_time_formatted": "00:01:00.000"}
```

## Documentation

- **[Stdin Modes](docs/stdin-modes.md)** - WAV stdin, raw PCM stdin, and multiplexed stdin (IPC) with code examples for Node.js, Python, Go
- **[Development](docs/development.md)** - Type checking, linting, testing, Docker, detection algorithm details

## Development

```shell
uv run basedpyright  # Type checking
uv run ruff check    # Linting
uv run pytest        # Testing
```

See [docs/development.md](docs/development.md) for more details.
