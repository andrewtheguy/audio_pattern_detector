//! Detects audio patterns (intros, breaks, station beeps) in audio files and streams.
//!
//! Detection is a two-step process: FFT cross-correlation finds candidate
//! locations, then each candidate is verified by correlation-envelope shape
//! (normal and short clips) or by a narrowband spectral check (marker tones
//! declared in `.apd.toml` pattern configs). See `docs/pattern-matching.md`.

pub mod audio_clip;
pub mod detector;
pub mod dsp;
pub mod error;
pub mod ffmpeg;
pub mod matching;
pub mod pattern_config;
#[cfg(feature = "python")]
mod python;
pub mod stream;
pub mod time_format;
pub mod tone;
pub mod wav;

pub use audio_clip::{AudioClip, MarkerToneParams, MarkerToneThresholds, Strategy, DEFAULT_TARGET_SAMPLE_RATE};
pub use detector::{
    AudioPatternDetector, ClipConfig, DetectorConfig, DetectorOptions, PatternDetectedCallback, PeakTimes,
    DEFAULT_SECONDS_PER_CHUNK,
};
pub use error::{Error, Result};
pub use matching::{
    find_clips_in_file, find_clips_in_wav_stream, load_pattern_clips, match_pattern, match_pattern_multiplexed,
    match_pattern_wav_stream, read_multiplexed_patterns, MatchOptions,
};
pub use stream::{AudioStream, SampleSource};
