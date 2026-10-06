//! High-level entry points: load patterns, open the audio, run the detector.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::audio_clip::{AudioClip, DEFAULT_TARGET_SAMPLE_RATE};
use crate::detector::{
    AudioPatternDetector, DetectorOptions, PatternDetectedCallback, PeakTimes, DEFAULT_SECONDS_PER_CHUNK,
};
use crate::error::{Error, Result};
use crate::ffmpeg::FfmpegSource;
use crate::stream::{AudioStream, WavFileSource, WavStreamSource};
use crate::wav::read_full;

const MAX_MULTIPLEXED_PATTERNS: u32 = 100;
const MAX_PATTERN_NAME_BYTES: u32 = 1024;
const MAX_PATTERN_DATA_BYTES: u32 = 100 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct MatchOptions {
    pub debug_mode: bool,
    /// Seconds per chunk for the sliding window (`None` to auto-compute).
    pub seconds_per_chunk: Option<u32>,
    /// Sample rate used for processing; patterns and WAV files are resampled to it.
    pub target_sample_rate: u32,
    pub debug_dir: PathBuf,
    /// Override the minimum correlation peak height (default: 0.25).
    pub height_min: Option<f32>,
}

impl Default for MatchOptions {
    fn default() -> Self {
        Self {
            debug_mode: false,
            seconds_per_chunk: Some(DEFAULT_SECONDS_PER_CHUNK),
            target_sample_rate: DEFAULT_TARGET_SAMPLE_RATE,
            debug_dir: PathBuf::from("./tmp"),
            height_min: None,
        }
    }
}

impl MatchOptions {
    fn detector_options(&self) -> DetectorOptions {
        DetectorOptions {
            debug_mode: self.debug_mode,
            seconds_per_chunk: self.seconds_per_chunk,
            target_sample_rate: self.target_sample_rate,
            debug_dir: self.debug_dir.clone(),
            height_min: self.height_min,
        }
    }
}

/// Load pattern clips from `.wav` / `.apd.toml` (or ffmpeg-decodable) files.
pub fn load_pattern_clips<P: AsRef<Path>>(pattern_files: &[P], sample_rate: u32) -> Result<Vec<AudioClip>> {
    let mut pattern_clips = Vec::with_capacity(pattern_files.len());
    let mut clip_names_seen: HashMap<String, PathBuf> = HashMap::new();
    for pattern_file in pattern_files {
        let pattern_file = pattern_file.as_ref();
        if !pattern_file.exists() {
            return Err(Error::invalid(format!("Pattern {} does not exist", pattern_file.display())));
        }
        let pattern_clip = AudioClip::from_audio_file(pattern_file, sample_rate)?;
        if let Some(previous) = clip_names_seen.get(&pattern_clip.name) {
            return Err(Error::invalid(format!(
                "Duplicate clip name '{}' from files:\n  - {}\n  - {}\nRename one of the files so clip names are unique.",
                pattern_clip.name,
                previous.display(),
                pattern_file.display()
            )));
        }
        clip_names_seen.insert(pattern_clip.name.clone(), pattern_file.to_path_buf());
        pattern_clips.push(pattern_clip);
    }

    if pattern_clips.is_empty() {
        return Err(Error::invalid("No pattern clips passed"));
    }
    Ok(pattern_clips)
}

/// Find pattern matches in an audio file.
///
/// WAV files are decoded natively (mixed to mono and resampled as needed);
/// other formats are decoded through ffmpeg.
///
/// Returns `(peak times per clip or None if accumulate_results is false,
/// total seconds processed)`.
pub fn match_pattern<P: AsRef<Path>>(
    audio_source: impl AsRef<Path>,
    pattern_files: &[P],
    options: &MatchOptions,
    on_pattern_detected: Option<PatternDetectedCallback>,
    accumulate_results: bool,
) -> Result<(Option<PeakTimes>, f64)> {
    let audio_source = audio_source.as_ref();
    if !audio_source.exists() {
        return Err(Error::invalid(format!("Audio {} does not exist", audio_source.display())));
    }
    let pattern_clips = load_pattern_clips(pattern_files, options.target_sample_rate)?;
    let detector = AudioPatternDetector::new(pattern_clips, options.detector_options())?;
    find_clips_in_file(&detector, audio_source, on_pattern_detected, accumulate_results)
}

/// Run an already constructed detector over an audio file (see [`match_pattern`]).
pub fn find_clips_in_file(
    detector: &AudioPatternDetector,
    audio_source: impl AsRef<Path>,
    on_pattern_detected: Option<PatternDetectedCallback>,
    accumulate_results: bool,
) -> Result<(Option<PeakTimes>, f64)> {
    let audio_source = audio_source.as_ref();
    if !audio_source.exists() {
        return Err(Error::invalid(format!("Audio {} does not exist", audio_source.display())));
    }
    let sr = detector.target_sample_rate();

    let audio_name = audio_source
        .file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    eprintln!("Finding pattern in audio file {audio_name}...");

    let is_wav = audio_source
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));

    if is_wav {
        let source = WavFileSource::open(audio_source, sr)?;
        let mut stream = AudioStream::new(audio_name, source, sr);
        return detector.find_clip_in_audio(&mut stream, on_pattern_detected, accumulate_results);
    }

    let mut source = FfmpegSource::open(audio_source, sr)?;
    let result = {
        let mut stream = AudioStream::new(audio_name, &mut source, sr);
        detector.find_clip_in_audio(&mut stream, on_pattern_detected, accumulate_results)?
    };
    source.finish()?;
    Ok(result)
}

/// Run an already constructed detector over a WAV stream. The WAV must be
/// mono at the detector's sample rate.
pub fn find_clips_in_wav_stream<R: Read>(
    detector: &AudioPatternDetector,
    reader: R,
    on_pattern_detected: Option<PatternDetectedCallback>,
    accumulate_results: bool,
) -> Result<(Option<PeakTimes>, f64)> {
    let sr = detector.target_sample_rate();
    let source = WavStreamSource::new(reader, sr)?;
    eprintln!("WAV stdin: {sr}Hz, mono, {}", source.format().name());

    let mut stream = AudioStream::new("stdin", source, sr);
    detector.find_clip_in_audio(&mut stream, on_pattern_detected, accumulate_results)
}

fn find_in_wav_stream<R: Read>(
    reader: R,
    pattern_clips: Vec<AudioClip>,
    options: &MatchOptions,
    on_pattern_detected: Option<PatternDetectedCallback>,
    accumulate_results: bool,
) -> Result<(Option<PeakTimes>, f64)> {
    let detector = AudioPatternDetector::new(pattern_clips, options.detector_options())?;
    find_clips_in_wav_stream(&detector, reader, on_pattern_detected, accumulate_results)
}

/// Find pattern matches in a WAV stream (e.g. stdin). The WAV must be mono
/// at the target sample rate.
pub fn match_pattern_wav_stream<R: Read, P: AsRef<Path>>(
    reader: R,
    pattern_files: &[P],
    options: &MatchOptions,
    on_pattern_detected: Option<PatternDetectedCallback>,
    accumulate_results: bool,
) -> Result<(Option<PeakTimes>, f64)> {
    let pattern_clips = load_pattern_clips(pattern_files, options.target_sample_rate)?;
    eprintln!("Finding pattern in audio stream stdin...");
    find_in_wav_stream(reader, pattern_clips, options, on_pattern_detected, accumulate_results)
}

fn read_uint32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    let got = read_full(reader, &mut buf)?;
    if got < 4 {
        return Err(Error::invalid(format!("Unexpected EOF reading uint32 (got {got} bytes)")));
    }
    Ok(u32::from_le_bytes(buf))
}

/// Read pattern clips from the multiplexed stream protocol.
///
/// Protocol format (all integers are uint32 little-endian):
///
/// ```text
/// [4 bytes] number_of_patterns
/// For each pattern:
///     [4 bytes] name_length
///     [name_length bytes] name (UTF-8)
///     [4 bytes] data_length
///     [data_length bytes] WAV data
/// ```
///
/// After the patterns are read, the rest of the stream contains audio data.
pub fn read_multiplexed_patterns<R: Read>(reader: &mut R, target_sample_rate: u32) -> Result<Vec<AudioClip>> {
    let num_patterns = read_uint32(reader)?;
    if num_patterns == 0 {
        return Err(Error::invalid("No patterns provided in multiplexed stdin"));
    }
    if num_patterns > MAX_MULTIPLEXED_PATTERNS {
        return Err(Error::invalid(format!(
            "Too many patterns ({num_patterns}), max is {MAX_MULTIPLEXED_PATTERNS}"
        )));
    }
    eprintln!("Reading {num_patterns} pattern(s) from multiplexed stdin...");

    let mut pattern_clips = Vec::with_capacity(num_patterns as usize);
    for i in 0..num_patterns {
        let name_length = read_uint32(reader)?;
        if name_length == 0 || name_length > MAX_PATTERN_NAME_BYTES {
            return Err(Error::invalid(format!("Invalid pattern name length: {name_length}")));
        }
        let mut name_bytes = vec![0u8; name_length as usize];
        if read_full(reader, &mut name_bytes)? < name_bytes.len() {
            return Err(Error::invalid(format!("Unexpected EOF reading pattern name {}", i + 1)));
        }
        let name = String::from_utf8(name_bytes)
            .map_err(|e| Error::invalid(format!("Pattern name {} is not valid UTF-8: {e}", i + 1)))?;

        let data_length = read_uint32(reader)?;
        if data_length == 0 {
            return Err(Error::invalid(format!("Pattern '{name}' has zero-length data")));
        }
        if data_length > MAX_PATTERN_DATA_BYTES {
            return Err(Error::invalid(format!("Pattern '{name}' data too large: {data_length} bytes")));
        }
        let mut wav_data = vec![0u8; data_length as usize];
        if read_full(reader, &mut wav_data)? < wav_data.len() {
            return Err(Error::invalid(format!("Unexpected EOF reading pattern '{name}' data")));
        }

        let clip = AudioClip::from_wav_bytes(&wav_data, &name, target_sample_rate)?;
        eprintln!("  Loaded pattern '{name}' ({:.2}s)", clip.clip_length_seconds());
        pattern_clips.push(clip);
    }
    Ok(pattern_clips)
}

/// Find pattern matches in a multiplexed stream: patterns first (see
/// [`read_multiplexed_patterns`]), then WAV audio until EOF.
pub fn match_pattern_multiplexed<R: Read>(
    mut reader: R,
    options: &MatchOptions,
    on_pattern_detected: Option<PatternDetectedCallback>,
    accumulate_results: bool,
) -> Result<(Option<PeakTimes>, f64)> {
    let pattern_clips = read_multiplexed_patterns(&mut reader, options.target_sample_rate)?;
    eprintln!("Reading WAV audio from stdin...");
    find_in_wav_stream(reader, pattern_clips, options, on_pattern_detected, accumulate_results)
}
