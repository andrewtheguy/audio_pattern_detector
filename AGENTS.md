strict no backward compatibility no matter what, so feel free to make breaking changes as needed. Just make sure to
do `cargo clippy --all-targets -- -D warnings` after changes to make sure code style is correct and then `cargo test` as needed.

- Use `./tmp` as the temporary working directory for debug output, scratch files, etc. It is gitignored.
- The project is a Cargo workspace: `crates/core` (`audio-pattern-detector-core`, the library) and `crates/cli` (`audio-pattern-detector`, the binary). Keep CLI-only dependencies (e.g. clap) out of the core crate. See `docs/development.md` for the code layout.
- Python bindings (PyO3) live in `crates/core/src/python.rs` behind the core crate's `python` feature, with type stubs in `audio_pattern_detector.pyi`; keep both in sync. When touching them also run `cargo clippy -p audio-pattern-detector-core --all-targets --features python -- -D warnings` and the tests in `tests/test_python_bindings.py`. See `docs/python.md`.

## Numerical code

- Numerical routines live in `crates/core/src/dsp/`. Reusable ones belong in their own crate: FFT cross-correlation is meant to come from the `fft-correlation` crate, so fix and optimise it there rather than in a local copy.
- They follow scipy/numpy/pyloudnorm semantics (`find_peaks`, `resample`, `correlate`, `hanning`, BS.1770 loudness) because the detection thresholds were tuned against those. Keep that behaviour when touching them.
- Rounding is the exception: use Rust's standard `f64::round` (ties away from zero), not Python's round-half-to-even.

## Testing

- Use test data constants at the top of test files (e.g. `RAINBOW_INTRO_PATTERN`, `RAINBOW_INTRO_AUDIO`) instead of hardcoding paths — makes swapping clips a one-line change.
- Tests should assert exact expected values, not just lengths. For example, assert the full output vector, not just `out.len() == 5`.
- Shared integration-test helpers go in `crates/core/tests/common/mod.rs`. CLI tests live in `crates/cli/tests/`.
- Sample audio files go in the repo-root `sample_audios/` (clips in `sample_audios/clips/`); Rust tests reference them as `../../sample_audios/...`. Keep them small (~30s audio sections).
- When a clip produces false positives or is too short to work reliably, replace it rather than loosening thresholds.

## Debugging opus/lossy audio

- Use `--debug` and `--debug-dir <dir>` to save debug output per run. Use separate debug dirs for A/B comparisons.
- Debug graphs are not implemented yet (planned for v2, see `docs/roadmap.md`). Until then use the stderr diagnostics and the per-section peak dumps in `debug/cross_correlation_<clip>/`, which contain the MSE similarity and per-window Pearson r of every candidate.
- Debug audio sections in `audio_section/` can be listened to for ground truth verification. Don't assume old detection results are correct.
- Pattern clips should be extracted from the same encoding as the target audio (e.g. from an Opus stream when matching Opus audio). Denoise as a fallback when source-matched clips aren't available. See `docs/denoise-strategy.md`.

## Version bumping

- Bump `[workspace.package] version` in the root `Cargo.toml` (both crates inherit it), then run `cargo build` to update `Cargo.lock`. The Python wheel takes its version from it too.
