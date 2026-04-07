# Native Helper (Rust)

## Why Rust instead of scipy

The `native-helper/` crate provides Rust implementations of numerical routines (FFT cross-correlation, Pearson correlation, peak finding, BS.1770 loudness, resampling, Simpson integration) exposed to Python via PyO3. These replace what would otherwise come from scipy.

Two factors motivated this choice:

### 1. Dependency bloat

scipy is a large package (~40 MB installed) that pulls in a substantial dependency tree. This project only needs a handful of routines from it. Implementing them in a focused Rust crate keeps the dependency footprint small and the install fast.

### 2. Cold-start overhead in containers

scipy has significant warm-up cost on first use: lazy-loading shared libraries, JIT-compiling via internal dispatch, and importing large submodules. In containerized environments (Docker, serverless) where the process starts fresh on every invocation, this overhead is paid repeatedly and adds noticeable latency to each run. The standard workaround is pre-warming hacks (dummy imports at container build time, keeping containers alive longer, etc.), which add complexity. A compiled Rust extension has near-zero cold-start cost since all code is ahead-of-time compiled into a single shared library.

## scipy as a dev-only QA dependency

scipy is included in the root `dev` dependency group solely for QA: the comparison scripts in `native-helper/scripts/` and some binding tests use scipy as a reference implementation to verify the Rust routines produce correct results. It is never imported by production code and is not included in production Docker images.

## Structure

- `src/lib.rs` -- core Rust implementations (pure Rust, no Python dependency)
- `src/python.rs` -- PyO3 bindings that wrap the core functions for Python
- Type stubs: `native-helper/native_helper.pyi` and `typings/native_helper/__init__.pyi`

## Building

```shell
uv run maturin develop --manifest-path native-helper/Cargo.toml
```

## Testing

```shell
# Rust unit tests
cargo test --manifest-path native-helper/Cargo.toml

# Python binding tests
uv run pytest native-helper/tests/
```
