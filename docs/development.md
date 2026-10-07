# Development

## Building

```shell
cargo build            # debug build of the whole workspace
cargo build --release  # optimized binary at target/release/audio-pattern-detector
```

## Linting

```shell
cargo clippy --all-targets -- -D warnings
```

## Testing

```shell
cargo test
```

Unit tests live next to the code in each crate's `src/`. Integration tests in `crates/core/tests/` run the library, and `crates/cli/tests/` runs the CLI binary, against the clips in the repo-root `sample_audios/` (referenced as `../../sample_audios/...`, since cargo runs integration tests from the crate directory). Run the tests in release mode (`cargo test --release`) if the FFT-heavy integration tests feel slow.

The Python bindings have their own checks, see [python.md](python.md):

```shell
cargo clippy -p audio-pattern-detector-core --all-targets --features python -- -D warnings
cargo build -p audio-pattern-detector-core --features python --lib && python -m unittest tests.test_python_bindings
```

## Code layout

The repo is a Cargo workspace with two crates:

- `crates/core` — `audio-pattern-detector-core` (library `audio_pattern_detector_core`): the detector, audio I/O and the Python bindings. No CLI dependencies.
- `crates/cli` — `audio-pattern-detector`: the `audio-pattern-detector` binary, a thin clap/JSONL front end over the core crate.

The version is shared through `[workspace.package]` in the root `Cargo.toml`.

| Path | Contents |
|------|----------|
| `crates/cli/src/main.rs` | CLI (`match`, `show-config`) and JSONL output |
| `crates/core/src/matching.rs` | High-level entry points: file, WAV stream and multiplexed stream matching |
| `crates/core/src/detector.rs` | `AudioPatternDetector`: chunking, Step 1 correlation, Step 2 verification |
| `crates/core/src/tone.rs` | Pure-tone analysis for the marker-tone strategy |
| `crates/core/src/pattern_config.rs` | `.apd.toml` loader |
| `crates/core/src/audio_clip.rs` | `AudioClip` and verification strategies |
| `crates/core/src/stream.rs` | Audio sources (`SampleSource`): memory, raw float32, WAV file, WAV stream |
| `crates/core/src/wav.rs` | WAV reading and writing |
| `crates/core/src/ffmpeg.rs` | ffmpeg subprocess for non-WAV files |
| `crates/core/src/python.rs` | Python bindings (PyO3), compiled only with the `python` feature; see [python.md](python.md) |
| `crates/core/src/dsp/` | BS.1770 loudness, peak finding, resampling, Pearson correlation, spectra |

FFT cross-correlation comes from the [`fft-correlation`](https://github.com/andrewtheguy/fft-correlation) crate; fixes and optimizations to it belong there. The numerical routines in `crates/core/src/dsp/` and that crate follow the semantics of the scipy/numpy/pyloudnorm functions the algorithm was originally tuned against (`scipy.signal.find_peaks`, `scipy.signal.resample`, `scipy.signal.correlate`, `numpy.hanning`, BS.1770 integrated loudness), so thresholds carry over unchanged.

## Debug output

There is no debug mode at the moment; it is being rebuilt from scratch. See [roadmap.md](roadmap.md) for the plan.

## Detection Algorithm

Detection is a **two-step process**:

- **Step 1 — Candidate detection**: FFT cross-correlation against the audio section, followed by peak detection (height ≥ 0.25, min distance = clip length). This step always runs first for every clip type and produces centered candidate match locations.
- **Step 2 — Candidate verification**: each candidate from Step 1 is verified by exactly one of three branches, chosen by clip type. The branches below are alternatives — they all share Step 1.

### Step 2 paths

**Normal Patterns (`.wav` clips ≥ 0.5s)** — verification uses partitioned MSE plus multi-window Pearson correlation. The cross-correlation curve is downsampled across 3 overlapping regions (0-50%, 40-60%, 50-100%) and compared against the clip's self-correlation. Pearson r is scale-invariant, making it robust against lossy codec artifacts (Opus, AAC) that inflate the correlation envelope but preserve shape. High Pearson r (≥ 0.90) can override moderate MSE, allowing detection even with degraded audio. See [docs/denoise-strategy.md](denoise-strategy.md) for improving pattern clip quality. Works well with repeating or non-repeating patterns that are loud enough within the audio section because the correlation is normalized against the clip's own self-correlation peak, which helps eliminate false positives that are much softer or unrelated.

**Short Clips (`.wav` clips < 0.5s)** — uses the same correlation-envelope approach as Normal Patterns but with simplified windowing: a single 0-100% Pearson window and whole-only MSE (no middle partition emphasis), since the correlation envelope is too short for sub-region analysis. Short clips must cross-correlate well at Step 1 — this is the user's responsibility when providing the clip.

**Marker Tone (`.apd.toml` clips with `strategy = "marker_tone"`)** — used for things like the RTHK hourly station beep, where a clean sine does not produce a distinctive enough Step-1 envelope for the shape-based paths to verify reliably. Step 1 still runs (cross-correlation against the synthesised tone clip + peak detection); Step 2 substitutes a narrowband spectral check at the declared dominant frequency. Each `.apd.toml` file can carry its own `verification` thresholds, so stations like RTHK, 881, and 903 can tune the same verifier without separate strategy code. This is independent of the short clip path. See [docs/pattern-matching.md](pattern-matching.md) for the `.apd.toml` format.

The Normal/Short paths will miss distorted patterns like this because error score is too high and Pearson r is too low:

![rthk_beep_39_00:39:00_478782](https://github.com/user-attachments/assets/80669708-b8f9-461c-ae6c-2edddb161904)
