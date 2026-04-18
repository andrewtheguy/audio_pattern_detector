no backward compatibility, so feel free to make breaking changes as needed. Just make sure to
do `uv run basedpyright` `uv run ruff check` after changes to make sure code style is correct and then `uv run pytest` as needed.

- Use `./tmp` as the temporary working directory for debug output, scratch files, etc. It is gitignored.
- Always use uv for python commands to avoid trampling or breaking the system python environment.

## Native helper (Rust)

- Rust code lives in `native-helper/`. After changing Rust source, rebuild with `uv run maturin develop --skip-install --manifest-path native-helper/Cargo.toml`.
- Run Rust tests with `cargo test --manifest-path native-helper/Cargo.toml`.
- Run Python binding tests with `uv run pytest native-helper/tests/`.
- Follow the existing PyO3 binding pattern in `src/python.rs`: use `extract_f32_data`/`extract_f64_data` helpers, register in `__all__` and `add_function`.
- Update both type stubs: `native-helper/native_helper.pyi` and `typings/native_helper/__init__.pyi`.
- Prefer implementing computationally heavy algorithms in Rust even if they seem simple (e.g. Pearson correlation) — the user prefers Rust over scipy dependencies. See `docs/native-helper.md` for rationale.
- native-helper has no separate `pyproject.toml` — it is built by the root project's maturin config. Comparison scripts in `native-helper/scripts/` use scipy/pyloudnorm from the root dev group.

## Testing

- Use test data constants at the top of test files (e.g. `RAINBOW_INTRO_PATTERN`, `RAINBOW_INTRO_AUDIO`) instead of hardcoding paths — makes swapping clips a one-line change.
- Tests should assert exact expected values, not just lengths. For example, assert the full output array, not just `len(out) == 5`.
- Sample audio files go in `sample_audios/` (clips in `sample_audios/clips/`). Keep them small (~30s audio sections).
- When a clip produces false positives or is too short to work reliably, replace it rather than loosening thresholds.

## Debugging opus/lossy audio

- Debug graphs require matplotlib, which is not installed by default. Run `uv sync --group debug` to enable.
- Use `--debug` and `--debug-dir <dir>` to save graphs per run. Use separate debug dirs for A/B comparisons.
- The Pearson downsampled graphs in `graph/pearson_downsampled/` show the 101-point curves actually used for verification — check these first when troubleshooting.
- Debug audio sections in `audio_section/` can be listened to for ground truth verification. Don't assume old detection results are correct.
- Pattern clips should be extracted from the same encoding as the target audio (e.g. from an Opus stream when matching Opus audio). Denoise as a fallback when source-matched clips aren't available. See `docs/denoise-strategy.md`.

## Version bumping

- Bump patch version in `pyproject.toml` only, then run `uv lock` to update the lockfile. The `native-helper/Cargo.toml` version is permanently "0.0.0" (internal-only crate, never published).
