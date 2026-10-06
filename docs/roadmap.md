# Roadmap

## v1 — Rust rewrite, no charts (current, released as 0.4.0)

The detector is a single Rust crate (library + `audio-pattern-detector` binary), with optional Python bindings ([python.md](python.md)). Detection results match the previous Python implementation on all sample audio.

`--debug` is text and audio only:

| Output | Location under `--debug-dir` |
|--------|------------------------------|
| Per-candidate diagnostics (similarity, Pearson r, marker-tone metrics, rejection reason) | stderr |
| Audio around each candidate, for listening | `audio_section/<clip>/<clip>_<index>_<section_ts>_<peak>.wav` |
| Peak dump: candidate peaks, their times, MSE and per-window Pearson r | `debug/cross_correlation_<clip>/<index>_<section_ts>.txt` (JSON) |

Not in v1: any PNG graph output. The Python version drew these with matplotlib.

## v2 — simple debug charts

Goal: bring back only the charts that are needed to tune detection, with only the chart elements needed to read them. Charts stay a debug aid: opt-in at build time, written to files, never displayed.

### Charts

| Chart | File under `--debug-dir` | What it answers |
|-------|--------------------------|-----------------|
| Pearson windows | `graph/pearson_downsampled/<clip>/<clip>_<index>_<section_ts>_<peak>_w<l>_<r>.png` | The downsampled candidate window against the pattern's window — the curves verification actually compares. Check this first. |
| Correlation slice | `graph/cross_correlation_slice/<clip>/<clip>_<index>_<section_ts>_<peak>.png` | The full-resolution candidate envelope against the pattern's self-correlation: where does the shape diverge? |
| Section correlation | `graph/cross_correlation/<clip>/<clip>_<index>_<section_ts>.png` | The whole section's correlation curve: were there candidate peaks at all, and how high? |

The first two apply to the correlation-envelope paths (normal and short clips) and, as before, are only drawn for candidates with similarity ≤ 0.1. The third applies to every clip, including marker tones.

Dropped from the Python version:

- `clip_correlation` and `cross_correlation_slice_original` — the pattern's self-correlation on its own; the same curve is already overlaid in the correlation slice chart.
- `mean_squared_error_similarity` scatter — the values are in the peak dump.

### Chart elements

- Line series only: candidate in one colour, pattern overlaid in a second, semi-transparent colour.
- Title (carrying the Pearson r and the `*best*` window marker where relevant).
- X and Y axes with tick labels and an axis label each.
- Fixed 1000×400 px PNG.

Not included: legends, grid lines, themes, annotations, interactivity, configurable sizes or formats.

### Implementation notes

- Library: [`plotters`](https://crates.io/crates/plotters) with the bitmap backend only (`default-features = false`).
- Text needs a font. Use the `ab_glyph` feature with a font embedded via `include_bytes!` so the binary has no fontconfig/freetype dependency and renders identically everywhere.
- Gate everything behind an optional `charts` cargo feature so default builds do not carry the dependency. Without the feature `--debug` behaves as in v1.
- The detector already has the data at the right places: `verify_correlation_envelope` holds the slice, the pattern curve and the downsampled windows; `correlation_method` holds the section correlation.
- The `:` in `<section_ts>` (`00:39:00`) is not a valid filename character on Windows. Pick a portable timestamp form when charts land, and apply it to the v1 debug files at the same time.

### Open questions

- Debug mode is still switched off unless `--chunk-seconds` is 60 (inherited from the Python version). Nothing in the v1 output depends on that; decide whether to lift it together with the charts.
- Whether marker-tone candidates deserve a chart of their own (e.g. per-frame band purity). Not planned until a tuning session shows the stderr metrics are not enough.
