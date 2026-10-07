# Audio Pattern Detector

Detects audio patterns specified by audio clips in target audio files. Designed to detect intros, breaks, and outros from prerecorded radio shows and podcasts.

Useful for AI workflows to efficiently segment audio files before processing (e.g., OpenAI Whisper transcription preprocessing).

Detection is a two-step process. **Step 1** always runs FFT cross-correlation against the audio to find and center potential match locations. **Step 2** verifies each candidate using one of three paths chosen by clip type: normal verification (partitioned MSE + center-window Pearson correlation), short-clip verification (single-window variant for clips under 0.5s), or marker-tone verification (narrowband spectral check for `.apd.toml` patterns like station beeps). Robust against lossy-encoded audio (Opus, AAC).

Written in Rust: a single self-contained binary with no runtime dependencies (ffmpeg is only needed for non-WAV input files). Also available as a Python package with the same detector ([Python bindings](docs/python.md)).

## Installation

### Prebuilt binaries

Download the archive for your platform (Linux x86_64, Linux arm64, macOS Apple Silicon, Windows x86_64) from the [GitHub releases](https://github.com/andrewtheguy/audio_pattern_detector/releases) page and put `audio-pattern-detector` on your `PATH`. The Linux binaries and wheels are built on GitHub's current Ubuntu runners and need a glibc at least as new as theirs; older distributions are not supported.

### Install from source (requires Rust toolchain)

```shell
cargo install --git https://github.com/andrewtheguy/audio_pattern_detector.git --tag vx.x.x audio-pattern-detector
```

### Docker image

`ghcr.io/andrewtheguy/audio_pattern_detector:<tag>` is published with each release (`v<version>`, plus `latest` for releases from `main`). It contains only the binary at `/usr/local/bin/audio-pattern-detector` (no ffmpeg), so the usual use is copying it into another image:

```dockerfile
COPY --from=ghcr.io/andrewtheguy/audio_pattern_detector:v<version> /usr/local/bin/audio-pattern-detector /usr/local/bin/audio-pattern-detector
```

The release workflow builds the image for amd64 and arm64 from the binaries it has already compiled (`runtime-prebuilt` target). To build it locally from source for the host architecture: `docker build --target runtime -t audio-pattern-detector .`

### Python package

```shell
pip install audio-pattern-detector --extra-index-url https://andrewtheguy.github.io/audio_pattern_detector/simple/
```

```python
import audio_pattern_detector as apd

detector = apd.Detector(["pattern.wav", "station_beep.apd.toml"])
result = detector.match_file("audio.wav")
print(result.detections)  # {"pattern": [12.345], "station_beep": []}
```

See [Python bindings](docs/python.md) for the API, streaming input and uv configuration.

### Run from a checkout

```shell
cargo run --release -- [command] [options]
```

## Audio Requirements

- **Mono**: detection runs on mono audio. WAV files with more channels are mixed down; stdin streams must already be mono.
- **Sample Rate**: Default is 8kHz (configurable via `--target-sample-rate`). WAV files and pattern clips at other rates are resampled; stdin streams must already be at the target rate.
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
| `--pattern-file`       | Single pattern file (`.wav` or `.apd.toml`), repeatable                  |
| `--pattern-folder`     | Folder of pattern clips (`*.wav` and `*.apd.toml`), repeatable           |
| `--chunk-seconds`      | Seconds per chunk (default: 60, or "auto")                               |
| `--timestamp-format`   | JSONL timestamp fields: `both` (default), `ms`, or `formatted`           |
| `--height-min`         | Minimum correlation peak height (default: 0.25; lower to find weak matches) |

## JSONL Output Format

Output is always streaming JSONL:

By default, JSONL timestamp events include both millisecond and formatted
fields. Use `--timestamp-format ms` or `--timestamp-format formatted` to emit
just one representation.

```jsonl
{"type":"start","source":"audio.wav"}
{"type":"pattern_detected","clip_name":"pattern","timestamp_ms":5500,"timestamp_formatted":"00:00:05.500"}
{"type":"end","total_time_ms":60000,"total_time_formatted":"00:01:00.000"}
```

Errors are reported on stderr as `Error: <message>` with exit code 1.

## Library

The crate can also be used as a Rust library:

```rust
use audio_pattern_detector::{match_pattern, MatchOptions};

let (peak_times, total_seconds) = match_pattern(
    "audio.wav",
    &["pattern.wav", "station_beep.apd.toml"],
    &MatchOptions::default(),
    Some(&mut |clip_name, seconds| println!("{clip_name} at {seconds:.3}s")),
    true,
)?;
```

For custom audio sources, implement `SampleSource` and drive `AudioPatternDetector::find_clip_in_audio` directly.

## Documentation

- **[Pattern Matching](docs/pattern-matching.md)** - Detailed description of the detection pipeline, verification logic, and thresholds
- **[Denoise Strategy](docs/denoise-strategy.md)** - How to denoise pattern clips for better matching with lossy-encoded or noisy audio
- **[Python Bindings](docs/python.md)** - Python API, installation and building the wheel
- **[Stdin Modes](docs/stdin-modes.md)** - WAV stdin and multiplexed stdin (IPC) with a Node.js example
- **[Development](docs/development.md)** - Building, linting, testing, code layout
- **[Roadmap](docs/roadmap.md)** - The plan for the debug mode and its charts in v2

## Development

```shell
cargo clippy --all-targets -- -D warnings  # Linting
cargo test                                 # Testing

# Python bindings (see docs/python.md)
cargo clippy -p audio-pattern-detector-core --all-targets --features python -- -D warnings
cargo build -p audio-pattern-detector-core --features python --lib && python -m unittest tests.test_python_bindings
```

See [docs/development.md](docs/development.md) for more details.
