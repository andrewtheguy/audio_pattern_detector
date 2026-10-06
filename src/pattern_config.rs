//! Loader for `.apd.toml` pattern config files.
//!
//! `.apd.toml` is a TOML document with two sections that mirror the detection
//! pipeline:
//!
//! * `[clip]` — Step 1 audio source used by FFT cross-correlation. The clip can
//!   be synthesised from a formula (`source = "sine"`) or carried inline as a
//!   base64-encoded WAV (`source = "wav_base64"`).
//! * `[verification]` — Step 2 verification logic. Declares the strategy
//!   (currently only `marker_tone`) and per-strategy thresholds.

use std::path::Path;

use base64::Engine;
use toml::{Table, Value};

use crate::audio_clip::{MarkerToneParams, MarkerToneThresholds, Strategy};
use crate::error::{Error, Result};
use crate::stream::resample_audio;
use crate::wav::load_wav_from_bytes;

pub const APD_EXTENSION: &str = ".apd.toml";

/// Strategies understood by the detector.
pub const VALID_STRATEGIES: &[&str] = &["marker_tone"];

/// Clip-source kinds understood by the loader.
pub const VALID_CLIP_SOURCES: &[&str] = &["sine", "wav_base64"];

// Per-source allowed fields (excluding `source` itself). All lists are sorted
// because they are printed in error messages.
const SINE_FIELDS: &[&str] = &["amplitude", "duration_seconds", "frequency_hz"];
const WAV_BASE64_FIELDS: &[&str] = &["data"];

pub const VALID_VERIFICATION_THRESHOLDS: &[&str] = &[
    "maximum_max_flank_purity",
    "maximum_min_flank_purity",
    "minimum_active_frame_mean_purity",
    "minimum_active_frame_ratio",
    "minimum_band_purity",
    "minimum_longest_active_run",
];

// All keys allowed inside the [verification] table.
const VERIFICATION_FIELDS: &[&str] = &[
    "dominant_frequency_hz",
    "maximum_max_flank_purity",
    "maximum_min_flank_purity",
    "minimum_active_frame_mean_purity",
    "minimum_active_frame_ratio",
    "minimum_band_purity",
    "minimum_longest_active_run",
    "strategy",
];

// Keys allowed at the top level of the document.
const TOP_LEVEL_FIELDS: &[&str] = &["clip", "description", "verification"];

/// Parsed `.apd.toml` file.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternConfig {
    pub strategy: Strategy,
    pub audio: Vec<f32>,
}

fn format_list<S: AsRef<str>>(items: &[S]) -> String {
    let quoted: Vec<String> = items.iter().map(|s| format!("'{}'", s.as_ref())).collect();
    format!("[{}]", quoted.join(", "))
}

fn unknown_fields(table: &Table, allowed: &[&str]) -> Vec<String> {
    let mut unknown: Vec<String> = table
        .keys()
        .filter(|key| !allowed.contains(&key.as_str()))
        .cloned()
        .collect();
    unknown.sort();
    unknown
}

fn get_required<'a>(table: &'a Table, key: &str, path: &str) -> Result<&'a Value> {
    table
        .get(key)
        .ok_or_else(|| Error::invalid(format!("{path}: missing required field '{key}'")))
}

fn type_error(path: &str, key: &str, expected: &str, value: &Value) -> Error {
    Error::invalid(format!(
        "{path}: field '{key}' must be {expected}, got {}",
        value.type_str()
    ))
}

fn get_table<'a>(table: &'a Table, key: &str, path: &str) -> Result<&'a Table> {
    let value = get_required(table, key, path)?;
    value.as_table().ok_or_else(|| type_error(path, key, "table", value))
}

fn get_str<'a>(table: &'a Table, key: &str, path: &str) -> Result<&'a str> {
    let value = get_required(table, key, path)?;
    value.as_str().ok_or_else(|| type_error(path, key, "string", value))
}

fn get_number(table: &Table, key: &str, path: &str) -> Result<f64> {
    let value = get_required(table, key, path)?;
    match value {
        Value::Integer(i) => Ok(*i as f64),
        // TOML allows `inf` and `nan`.
        Value::Float(f) if !f.is_finite() => {
            Err(Error::invalid(format!("{path}: '{key}' must be a finite number, got {f}")))
        }
        Value::Float(f) => Ok(*f),
        _ => Err(type_error(path, key, "integer/float", value)),
    }
}

/// Longest synthesised sine clip; bounds the allocation for a bad config.
const MAX_SINE_DURATION_SECONDS: f64 = 3600.0;

fn clip_from_sine(params: &Table, sample_rate: u32, source_path: &str) -> Result<Vec<f32>> {
    let unknown = unknown_fields(params, &[SINE_FIELDS, &["source"]].concat());
    if !unknown.is_empty() {
        return Err(Error::invalid(format!(
            "{source_path}: unknown [clip] field(s) for source='sine': {}. Valid fields: {}",
            format_list(&unknown),
            format_list(SINE_FIELDS)
        )));
    }
    let frequency_hz = get_number(params, "frequency_hz", source_path)?;
    let duration_seconds = get_number(params, "duration_seconds", source_path)?;
    let amplitude = if params.contains_key("amplitude") {
        get_number(params, "amplitude", source_path)?
    } else {
        0.9
    };
    if frequency_hz <= 0.0 {
        return Err(Error::invalid(format!(
            "{source_path}: frequency_hz must be positive, got {frequency_hz}"
        )));
    }
    if duration_seconds <= 0.0 {
        return Err(Error::invalid(format!(
            "{source_path}: duration_seconds must be positive, got {duration_seconds}"
        )));
    }
    if frequency_hz * 2.0 >= sample_rate as f64 {
        return Err(Error::invalid(format!(
            "{source_path}: frequency_hz {frequency_hz} exceeds Nyquist ({}) for sample_rate {sample_rate}",
            sample_rate as f64 / 2.0
        )));
    }
    if duration_seconds > MAX_SINE_DURATION_SECONDS {
        return Err(Error::invalid(format!(
            "{source_path}: duration_seconds must be at most {MAX_SINE_DURATION_SECONDS}, got {duration_seconds}"
        )));
    }
    let n_samples = (duration_seconds * sample_rate as f64).round() as usize;
    // Synthesised in float32 throughout so the clip is identical on every platform.
    let angular_frequency = (2.0 * std::f64::consts::PI * frequency_hz) as f32;
    let amplitude = amplitude as f32;
    Ok((0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            amplitude * (angular_frequency * t).sin()
        })
        .collect())
}

fn clip_from_wav_base64(params: &Table, sample_rate: u32, source_path: &str) -> Result<Vec<f32>> {
    let unknown = unknown_fields(params, &[WAV_BASE64_FIELDS, &["source"]].concat());
    if !unknown.is_empty() {
        return Err(Error::invalid(format!(
            "{source_path}: unknown [clip] field(s) for source='wav_base64': {}. Valid fields: {}",
            format_list(&unknown),
            format_list(WAV_BASE64_FIELDS)
        )));
    }
    let data_str = get_str(params, "data", source_path)?;
    // Strip whitespace so callers can use TOML triple-quoted strings
    // (`data = """..."""`) and break the base64 across multiple lines.
    let cleaned: String = data_str.split_whitespace().collect();
    let wav_bytes = base64::engine::general_purpose::STANDARD
        .decode(cleaned)
        .map_err(|e| Error::invalid(format!("{source_path}: invalid base64 in [clip].data: {e}")))?;

    let (audio, source_sr) = load_wav_from_bytes(&wav_bytes, source_path)?;
    Ok(resample_audio(audio, source_sr, sample_rate))
}

fn parse_thresholds(verification: &Table, source_path: &str) -> Result<MarkerToneThresholds> {
    let number = |key: &str| -> Result<Option<f64>> {
        if verification.contains_key(key) {
            get_number(verification, key, source_path).map(Some)
        } else {
            Ok(None)
        }
    };
    let minimum_longest_active_run = match verification.get("minimum_longest_active_run") {
        None => None,
        Some(Value::Integer(i)) if *i >= 0 => Some(*i as usize),
        Some(value) => {
            return Err(type_error(source_path, "minimum_longest_active_run", "non-negative integer", value))
        }
    };
    Ok(MarkerToneThresholds {
        minimum_band_purity: number("minimum_band_purity")?,
        minimum_active_frame_ratio: number("minimum_active_frame_ratio")?,
        minimum_longest_active_run,
        minimum_active_frame_mean_purity: number("minimum_active_frame_mean_purity")?,
        maximum_min_flank_purity: number("maximum_min_flank_purity")?,
        maximum_max_flank_purity: number("maximum_max_flank_purity")?,
    })
}

/// Parse an `.apd.toml` file and return the clip audio + strategy metadata.
///
/// `sample_rate` is the target sample rate for the clip audio.
pub fn load_apd_file(path: impl AsRef<Path>, sample_rate: u32) -> Result<PatternConfig> {
    let path = path.as_ref();
    let source_path = path.display().to_string();
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::invalid(format!("{source_path}: failed to read file: {e}")))?;
    let doc: Table = text
        .parse()
        .map_err(|e| Error::invalid(format!("{source_path}: invalid TOML: {e}")))?;

    let unknown_top = unknown_fields(&doc, TOP_LEVEL_FIELDS);
    if !unknown_top.is_empty() {
        return Err(Error::invalid(format!(
            "{source_path}: unknown top-level field(s): {}. Valid fields: {} \
             (note: 'strategy' belongs in [verification])",
            format_list(&unknown_top),
            format_list(TOP_LEVEL_FIELDS)
        )));
    }

    let clip_section = get_table(&doc, "clip", &source_path)?;
    let source_kind = get_str(clip_section, "source", &source_path)?;
    let audio = match source_kind {
        "sine" => clip_from_sine(clip_section, sample_rate, &source_path)?,
        "wav_base64" => clip_from_wav_base64(clip_section, sample_rate, &source_path)?,
        other => {
            return Err(Error::invalid(format!(
                "{source_path}: unknown [clip].source '{other}'. Valid sources: {}",
                format_list(VALID_CLIP_SOURCES)
            )))
        }
    };

    let verification = get_table(&doc, "verification", &source_path)?;
    let unknown_v = unknown_fields(verification, VERIFICATION_FIELDS);
    if !unknown_v.is_empty() {
        return Err(Error::invalid(format!(
            "{source_path}: unknown [verification] field(s): {}. Valid fields: {}",
            format_list(&unknown_v),
            format_list(VERIFICATION_FIELDS)
        )));
    }

    let strategy_name = get_str(verification, "strategy", &source_path)?;
    if !VALID_STRATEGIES.contains(&strategy_name) {
        return Err(Error::invalid(format!(
            "{source_path}: unknown strategy '{strategy_name}'. Valid strategies: {}",
            format_list(VALID_STRATEGIES)
        )));
    }

    let dominant_frequency_hz = if verification.contains_key("dominant_frequency_hz") {
        Some(get_number(verification, "dominant_frequency_hz", &source_path)?)
    } else if source_kind == "sine" {
        // For sine clips the declared generator frequency is authoritative;
        // store it so the detector doesn't need to re-derive it from the
        // synthesised samples.
        Some(get_number(clip_section, "frequency_hz", &source_path)?)
    } else {
        // Leave unset; the detector falls back to deriving the frequency
        // from the loaded audio.
        None
    };

    Ok(PatternConfig {
        strategy: Strategy::MarkerTone(MarkerToneParams {
            dominant_frequency_hz,
            thresholds: parse_thresholds(verification, &source_path)?,
        }),
        audio,
    })
}
