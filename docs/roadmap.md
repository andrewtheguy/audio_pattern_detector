# Roadmap

## v2 — simple debug charts

There is no debug mode in the code at the moment; it is rebuilt from scratch in v2.

Goal: only the charts that are needed to tune detection, with only the chart elements needed to read them. Charts are a debug aid: opt-in at build time, written to files, never displayed.

### Charts

| Chart | File under the debug directory | What it answers |
|-------|--------------------------|-----------------|
| Pearson windows | `graph/pearson_downsampled/<clip>/<clip>_<index>_<section_ts>_<peak>_w<l>_<r>.png` | The downsampled candidate window against the pattern's window — the curves verification actually compares. Check this first. |
| Correlation slice | `graph/cross_correlation_slice/<clip>/<clip>_<index>_<section_ts>_<peak>.png` | The full-resolution candidate envelope against the pattern's self-correlation: where does the shape diverge? |
| Section correlation | `graph/cross_correlation/<clip>/<clip>_<index>_<section_ts>.png` | The whole section's correlation curve: were there candidate peaks at all, and how high? |

The first two apply to the correlation-envelope paths (normal and short clips) and are only drawn for candidates with similarity ≤ 0.1. The third applies to every clip, including marker tones.

Not planned:

- A chart of the pattern's self-correlation on its own — the same curve is already overlaid in the correlation slice chart.
- A scatter of the MSE similarities — plain numbers are enough for those.

### Chart elements

- Line series only: candidate in one colour, pattern overlaid in a second, semi-transparent colour.
- Title (carrying the Pearson r and the `*best*` window marker where relevant).
- X and Y axes with tick labels and an axis label each.
- Fixed 1000×400 px PNG.

Not included: legends, grid lines, themes, annotations, interactivity, configurable sizes or formats.

### Implementation notes

- Library: [`plotters`](https://crates.io/crates/plotters) with the bitmap backend only (`default-features = false`).
- Text needs a font. Use the `ab_glyph` feature with a font embedded via `include_bytes!` so the binary has no fontconfig/freetype dependency and renders identically everywhere.
- Gate the charts behind an optional `charts` cargo feature so default builds do not carry the dependency.
- The detector has the data at the right places: `verify_correlation_envelope` holds the slice, the pattern curve and the downsampled center window; `correlation_method` holds the section correlation.
- Debug file names write `<section_ts>` with `_` instead of `:` (`00_39_00`) so they are valid on Windows. Clip names can come from untrusted input (multiplexed stdin), so they must be sanitised to a single path component before being used in a file name.

### Open questions

- Which text and audio output the debug mode has besides the charts (per-candidate diagnostics, the audio around each candidate, peak dumps).
- Whether marker-tone candidates deserve a chart of their own (e.g. per-frame band purity). Not planned until a tuning session shows the stderr metrics are not enough.
