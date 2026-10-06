# Python bindings

The crate ships [PyO3](https://pyo3.rs) bindings behind the `python` cargo feature, packaged with [maturin](https://www.maturin.rs) as the `audio-pattern-detector` wheel. The wheel contains only the compiled extension: no Python dependencies, and one `abi3` wheel per platform works on every Python >= 3.9. ffmpeg is needed on `PATH` only for non-WAV input files.

## Installation

Wheels (Linux x86_64 and arm64, macOS Apple Silicon, Windows x86_64) are attached to each [GitHub release](https://github.com/andrewtheguy/audio_pattern_detector/releases) and listed in a package index on GitHub Pages. The Linux wheels are built on GitHub's current Ubuntu runners and need a glibc at least as new as theirs.

```toml
# pyproject.toml (uv)
[project]
dependencies = ["audio-pattern-detector==0.4.0"]

[[tool.uv.index]]
name = "audio-pattern-detector"
url = "https://andrewtheguy.github.io/audio_pattern_detector/simple/"
explicit = true

[tool.uv.sources]
audio-pattern-detector = { index = "audio-pattern-detector" }
```

```shell
# pip
pip install audio-pattern-detector --extra-index-url https://andrewtheguy.github.io/audio_pattern_detector/simple/
```

## Usage

```python
import subprocess
import audio_pattern_detector as apd

# Loads and prepares the pattern clips once; reuse it for many audio sources.
detector = apd.Detector(["intro.wav", "station_beep.apd.toml"])

result = detector.match_file("show.m4a")
result.detections        # {"intro": [12.345], "station_beep": []}  seconds, sorted
result.duration_seconds  # seconds of audio processed

# Streaming: any binary file-like object that yields a mono WAV at the
# detector's sample rate, read until EOF. Detections are reported as they
# are found.
ffmpeg = subprocess.Popen(
    ["ffmpeg", "-v", "error", "-i", url, "-f", "wav", "-ac", "1", "-ar", "8000", "pipe:"],
    stdout=subprocess.PIPE,
)
result = detector.match_wav_stream(
    ffmpeg.stdout,
    on_detected=lambda clip_name, seconds: print(clip_name, seconds),
)
```

## API

### `Detector(pattern_files, *, seconds_per_chunk=60, target_sample_rate=8000, height_min=None, debug=False, debug_dir="./tmp")`

Pattern files are `.wav` or `.apd.toml` paths (other formats are decoded through ffmpeg). The keyword arguments are the library equivalents of the CLI's `--chunk-seconds` (`None` is `auto`), `--target-sample-rate`, `--height-min`, `--debug` and `--debug-dir`.

| Member | Description |
|--------|-------------|
| `clip_names` | Clip names in the order the pattern files were given |
| `seconds_per_chunk`, `target_sample_rate` | Effective settings |
| `config()` | Computed configuration as a dict, the same data as `show-config` |
| `match_file(audio_file, *, on_detected=None)` | Match an audio file; WAV is decoded natively (mixed to mono, resampled), anything else through ffmpeg |
| `match_wav_stream(stream, *, on_detected=None)` | Match a WAV stream from an object with `read(n) -> bytes`; it must already be mono at the target sample rate |

Both match methods return a `MatchResult`:

| Attribute | Description |
|-----------|-------------|
| `detections` | `dict[str, list[float]]`: sorted detection times in seconds for every clip, in pattern file order; an empty list when the clip was not found |
| `duration_seconds` | Seconds of audio processed |

`on_detected(clip_name, timestamp_seconds)` is called for each detection as soon as its chunk has been processed. Timestamps are float seconds at full sample resolution; the CLI's JSONL output rounds the same values to milliseconds.

### Module-level

| Name | Description |
|------|-------------|
| `clip_name(pattern_file)` | Clip name the detector reports for a pattern file (`903_beep.apd.toml` → `903_beep`, `intro.wav` → `intro`) |
| `APD_EXTENSION` | `".apd.toml"` |
| `DEFAULT_TARGET_SAMPLE_RATE`, `DEFAULT_SECONDS_PER_CHUNK` | `8000`, `60` |
| `__version__` | Crate version |

### Behaviour

- Invalid input (missing files, bad WAV data, bad pattern configs, a chunk size too small for a clip) raises `ValueError`; I/O failures raise `OSError`.
- An exception raised by `on_detected` or by `stream.read()` is re-raised from the match call. A stream match stops reading as soon as that happens; a file match finishes the file first.
- The GIL is released while matching, so detectors can run in threads. A `Detector` is immutable and can be shared between threads.
- Diagnostics go to the process's stderr, as with the CLI.

## Building locally

```shell
# Build a wheel into tmp/wheels
uvx maturin build --release -o tmp/wheels

# Or install into the active virtualenv
uvx maturin develop --release

# Tests: against the installed module, or the extension built by cargo
cargo build --features python --lib
python -m unittest tests.test_python_bindings
```

The package version comes from `Cargo.toml`. Type stubs live in `audio_pattern_detector.pyi` at the repo root and are bundled into the wheel; update them together with `src/python.rs`.
