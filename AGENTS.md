no backward compatibility, so feel free to make breaking changes as needed. Just make sure to
do `uv run basedpyright` `uv run ruff check` after changes to make sure code style is correct and then `uv run pytest` as needed.

- Use `./tmp` as the temporary working directory for debug output, scratch files, etc. It is gitignored.

## Native helper (Rust)

- Rust code lives in `native-helper/`. After changing Rust source, rebuild with `uv run maturin develop --manifest-path native-helper/Cargo.toml`.
- Run Rust tests with `cargo test --manifest-path native-helper/Cargo.toml`.
- Run Python binding tests with `uv run pytest native-helper/tests/`.
- Follow the existing PyO3 binding pattern in `src/python.rs`: use `extract_f32_data`/`extract_f64_data` helpers, register in `__all__` and `add_function`.
- Update both type stubs: `native-helper/native_helper.pyi` and `typings/native_helper/__init__.pyi`.
- Prefer implementing computationally heavy algorithms in Rust even if they seem simple (e.g. Pearson correlation) — the user prefers Rust over scipy dependencies.

## Testing

- Use test data constants at the top of test files (e.g. `RAINBOW_INTRO_PATTERN`, `RAINBOW_INTRO_AUDIO`) instead of hardcoding paths — makes swapping clips a one-line change.
- Tests should assert exact expected values, not just lengths. For example, assert the full output array, not just `len(out) == 5`.
- Sample audio files go in `sample_audios/` (clips in `sample_audios/clips/`). Keep them small (~30s audio sections).
- When a clip produces false positives or is too short to work reliably, replace it rather than loosening thresholds.

## Debugging opus/lossy audio

- Use `--debug` and `--debug-dir <dir>` to save graphs per run. Use separate debug dirs for A/B comparisons.
- The Pearson downsampled graphs in `graph/pearson_downsampled/` show the 101-point curves actually used for verification — check these first when troubleshooting.
- Debug audio sections in `audio_section/` can be listened to for ground truth verification. Don't assume old detection results are correct.
- Pattern clips with background noise should be denoised for better matching on lossy streams. See `docs/denoise-strategy.md`.

## Version bumping

- Bump patch version in both `pyproject.toml` and `native-helper/Cargo.toml` together, then run `cargo check --manifest-path native-helper/Cargo.toml` and `uv lock` to update the lockfile.
