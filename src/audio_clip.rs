use std::path::Path;

use crate::error::Result;
use crate::ffmpeg::load_audio_ffmpeg;
use crate::pattern_config::{load_apd_file, APD_EXTENSION};
use crate::stream::resample_audio;
use crate::wav::{load_wav_file, load_wav_from_bytes};

/// Default sample rate for audio pattern detection (8kHz).
/// All audio clips and streams must use the same sample rate for matching to work.
pub const DEFAULT_TARGET_SAMPLE_RATE: u32 = 8000;

/// Per-clip overrides for the marker-tone verifier. Unset fields use the
/// detector defaults.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MarkerToneThresholds {
    pub minimum_band_purity: Option<f64>,
    pub minimum_active_frame_ratio: Option<f64>,
    pub minimum_longest_active_run: Option<usize>,
    pub minimum_active_frame_mean_purity: Option<f64>,
    pub maximum_min_flank_purity: Option<f64>,
    pub maximum_max_flank_purity: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MarkerToneParams {
    /// Expected tone frequency. When `None` the detector derives it from the
    /// clip audio, falling back to normal verification if it is not a pure tone.
    pub dominant_frequency_hz: Option<f64>,
    pub thresholds: MarkerToneThresholds,
}

/// Special verification strategy declared by an `.apd.toml` pattern config.
#[derive(Debug, Clone, PartialEq)]
pub enum Strategy {
    MarkerTone(MarkerToneParams),
}

impl Strategy {
    /// Name as written in `[verification].strategy`.
    pub fn name(&self) -> &'static str {
        match self {
            Strategy::MarkerTone(_) => "marker_tone",
        }
    }
}

/// A pattern to search for.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioClip {
    pub name: String,
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    /// `Some` when the clip was loaded from an `.apd.toml` pattern config.
    /// Drives strategy-based dispatch in the detector.
    pub strategy: Option<Strategy>,
}

impl AudioClip {
    pub fn new(name: impl Into<String>, audio: Vec<f32>, sample_rate: u32) -> Self {
        Self { name: name.into(), audio, sample_rate, strategy: None }
    }

    pub fn with_strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Clip name derived from a pattern file path: the file name without
    /// `.apd.toml` for pattern configs, otherwise without its last extension.
    pub fn name_for_path(clip_path: impl AsRef<Path>) -> String {
        let clip_path = clip_path.as_ref();
        let file_name = clip_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if file_name.to_lowercase().ends_with(APD_EXTENSION) {
            // Strip the full compound extension (e.g. "rthk_beep.apd.toml" -> "rthk_beep").
            return file_name[..file_name.len() - APD_EXTENSION.len()].to_string();
        }
        clip_path
            .file_stem()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// Load a clip from a file, resampled to `sample_rate`.
    ///
    /// Dispatches on extension: `.apd.toml` files are parsed as pattern
    /// configs, `.wav` files are decoded natively, anything else goes
    /// through ffmpeg.
    pub fn from_audio_file(clip_path: impl AsRef<Path>, sample_rate: u32) -> Result<Self> {
        let clip_path = clip_path.as_ref();
        let clip_name = Self::name_for_path(clip_path);
        let lower = clip_path
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        if lower.ends_with(APD_EXTENSION) {
            let config = load_apd_file(clip_path, sample_rate)?;
            return Ok(Self::new(clip_name, config.audio, sample_rate).with_strategy(config.strategy));
        }

        let audio = if lower.ends_with(".wav") {
            let (audio, source_sr) = load_wav_file(clip_path)?;
            resample_audio(audio, source_sr, sample_rate)
        } else {
            load_audio_ffmpeg(clip_path, sample_rate)?
        };
        Ok(Self::new(clip_name, audio, sample_rate))
    }

    /// Load a clip from WAV bytes, resampled to `sample_rate`.
    pub fn from_wav_bytes(wav_bytes: &[u8], name: &str, sample_rate: u32) -> Result<Self> {
        let (audio, source_sr) = load_wav_from_bytes(wav_bytes, name)?;
        Ok(Self::new(name, resample_audio(audio, source_sr, sample_rate), sample_rate))
    }

    pub fn clip_length_seconds(&self) -> f64 {
        self.audio.len() as f64 / self.sample_rate as f64
    }
}
