use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use serde_json::json;

use crate::audio_clip::{
    validate_sample_rate, AudioClip, MarkerToneThresholds, Strategy, DEFAULT_TARGET_SAMPLE_RATE,
};
use crate::dsp::{
    fft_correlate_full, find_peaks_1d, integrated_loudness, loudness_normalize, pearson_correlation_1d,
    resample_preserve_maxima_1d, CorrelationTemplate, CorrelationWorkspace, FindPeaksOptions,
};
use crate::error::{Error, Result};
use crate::stream::AudioStream;
use crate::time_format::seconds_to_time_whole;
use crate::tone::{
    analyze_pure_tone_candidate, extract_padded_segment, get_pure_tone_frequency, is_close, PureToneMetrics,
};
use crate::wav::write_wav_file;

/// Default seconds per chunk for sliding window processing.
pub const DEFAULT_SECONDS_PER_CHUNK: u32 = 60;

/// Clips shorter than this are verified with a single 0-100% window.
pub const SHORT_CLIP_DURATION_THRESHOLD: f64 = 0.5; // seconds

/// Default minimum height of a correlation peak to be considered a candidate.
pub const DEFAULT_HEIGHT_MIN: f32 = 0.25;

const TARGET_LUFS: f64 = -16.0;

// Correlation-envelope verification: 10 partitions, with the middle two
// checked separately because real distortions happen there most of the time.
const PARTITION_COUNT: usize = 10;
const MIDDLE_PARTITIONS: std::ops::Range<usize> = 4..6;
const SIMILARITY_HARD_LIMIT: f32 = 0.02;
const PEARSON_R_THRESHOLD: f64 = 0.90;
/// Downsampled points for a 20% window (2 partitions).
const PEARSON_DOWNSAMPLE_BASE: usize = 101;

/// Callback invoked as `on_pattern_detected(clip_name, timestamp_seconds)`.
pub type PatternDetectedCallback<'a> = &'a mut dyn FnMut(&str, f64);

/// Detected timestamps (seconds from the start of the audio) per clip name.
pub type PeakTimes = BTreeMap<String, Vec<f64>>;

#[derive(Debug, Clone)]
pub struct DetectorOptions {
    /// Enable debug output (diagnostics on stderr, candidate audio sections
    /// and peak dumps under `debug_dir`).
    pub debug_mode: bool,
    /// Seconds per chunk for sliding window processing. `None` or `0`
    /// auto-computes it as twice the longest clip (rounded up).
    pub seconds_per_chunk: Option<u32>,
    /// Sample rate of all clips and audio streams.
    pub target_sample_rate: u32,
    /// Base directory for debug output files.
    pub debug_dir: PathBuf,
    /// Override the minimum correlation peak height (default: 0.25).
    pub height_min: Option<f32>,
}

impl Default for DetectorOptions {
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

/// Configuration for a single clip in [`DetectorConfig`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ClipConfig {
    pub duration_seconds: f64,
    pub sliding_window_seconds: u32,
}

/// Computed configuration returned by [`AudioPatternDetector::get_config`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DetectorConfig {
    pub default_seconds_per_chunk: u32,
    pub min_chunk_size_seconds: u32,
    pub sample_rate: u32,
    /// Per-clip configuration in the order the clips were given.
    #[serde(serialize_with = "serialize_clip_configs")]
    pub clips: Vec<(String, ClipConfig)>,
}

impl DetectorConfig {
    pub fn clip(&self, name: &str) -> Option<&ClipConfig> {
        self.clips.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }
}

fn serialize_clip_configs<S: Serializer>(
    clips: &[(String, ClipConfig)],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    let mut map = serializer.serialize_map(Some(clips.len()))?;
    for (name, config) in clips {
        map.serialize_entry(name, config)?;
    }
    map.end()
}

/// Marker-tone verification settings resolved for one clip.
struct ToneVerifier {
    dominant_frequency: f64,
    thresholds: MarkerToneThresholds,
}

/// Pre-computed data for one clip.
struct ClipData {
    name: String,
    /// Loudness-normalized clip audio.
    clip: Vec<f32>,
    /// Seconds of the previous chunk prepended to each section: ceil(clip seconds).
    sliding_window: u32,
    /// Normalized absolute self-correlation: the "ideal" envelope shape.
    correlation_clip: Vec<f32>,
    correlation_clip_absolute_max: f32,
    /// `Some` when candidates are verified as a marker tone.
    tone: Option<ToneVerifier>,
    /// Downsampled windows of `correlation_clip` used for Pearson r.
    pearson_windows: Vec<Vec<f32>>,
}

/// Window of the correlation envelope compared with Pearson r, as
/// `(left partition, right partition, downsampled length)`.
type PearsonWindow = (usize, usize, usize);

fn pearson_window_specs(is_short_clip: bool) -> (&'static [PearsonWindow], usize) {
    // Lengths are proportional to window width: round(101 * partitions / 2).
    const SHORT: &[PearsonWindow] = &[(0, 10, 505)];
    const NORMAL: &[PearsonWindow] = &[(0, 5, 252), (4, 6, PEARSON_DOWNSAMPLE_BASE), (5, 10, 252)];
    if is_short_clip {
        (SHORT, 0)
    } else {
        (NORMAL, 1)
    }
}

fn downsample_window(curve: &[f32], window: PearsonWindow) -> Vec<f32> {
    let (left, right, samples) = window;
    let bound = |partition: usize| {
        ((curve.len() * partition) as f64 / PARTITION_COUNT as f64).round() as usize
    };
    resample_preserve_maxima_1d(&curve[bound(left)..bound(right)], samples)
}

fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::NAN;
    }
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum();
    (sum / a.len() as f64) as f32
}

fn mean(values: &[f32]) -> f32 {
    (values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64) as f32
}

/// Loudness-normalize audio in place to -16 dB LUFS. Silence comes out as NaN.
fn normalize_loudness(audio: &mut [f32], sample_rate: u32) {
    let seconds = audio.len() as f64 / sample_rate as f64;
    let block_size = if seconds < 0.5 { seconds } else { 0.4 };
    let loudness = integrated_loudness(audio, sample_rate, block_size);
    loudness_normalize(audio, loudness, TARGET_LUFS);
}

/// Replace every value with its absolute value and return the largest.
fn absolute_in_place(values: &mut [f32]) -> f32 {
    let mut absolute_max = f32::NEG_INFINITY;
    for v in values.iter_mut() {
        *v = v.abs();
        absolute_max = absolute_max.max(*v);
    }
    absolute_max
}

/// Absolute self-correlation normalized to a peak of 1, plus the original peak.
fn clip_correlation(clip: &[f32]) -> (Vec<f32>, f32) {
    let mut correlation = fft_correlate_full(clip, clip);
    let absolute_max = absolute_in_place(&mut correlation);
    correlation.iter_mut().for_each(|v| *v /= absolute_max);
    (correlation, absolute_max)
}

/// Clips whose sections are identical (same lookback), so one
/// loudness-normalized section and one forward FFT serve all of them.
struct SectionGroup {
    /// Seconds of the previous chunk prepended to every chunk after the
    /// first: ceil(clip seconds), the same for every clip in the group.
    lookback_seconds: u32,
    /// Longest clip in the group in samples; sizes the shared FFT.
    max_clip_length: usize,
    /// Indices into `AudioPatternDetector::clips`.
    clip_indices: Vec<usize>,
}

/// Group clips by sliding window, in order of first appearance.
fn section_groups(clips: &[ClipData]) -> Vec<SectionGroup> {
    let mut groups: Vec<SectionGroup> = Vec::new();
    for (clip_index, clip_data) in clips.iter().enumerate() {
        match groups.iter_mut().find(|g| g.lookback_seconds == clip_data.sliding_window) {
            Some(group) => {
                group.max_clip_length = group.max_clip_length.max(clip_data.clip.len());
                group.clip_indices.push(clip_index);
            }
            None => groups.push(SectionGroup {
                lookback_seconds: clip_data.sliding_window,
                max_clip_length: clip_data.clip.len(),
                clip_indices: vec![clip_index],
            }),
        }
    }
    groups
}

/// Buffers reused across the chunks of one [`AudioPatternDetector::find_clip_in_audio`] run.
struct RunBuffers {
    workspace: CorrelationWorkspace,
    /// One cached correlation template per clip, in clip order.
    templates: Vec<CorrelationTemplate>,
    audio_section: Vec<f32>,
    correlation: Vec<f32>,
}

/// Slice `width` samples centered on `middle_index`, zero-padding past either end.
pub fn slicing_with_zero_padding(array: &[f32], width: usize, middle_index: usize) -> Vec<f32> {
    let begin = middle_index as isize - (width / 2) as isize;
    extract_padded_segment(array, begin, width)
}

/// Per-candidate diagnostics dumped in debug mode.
#[derive(Default)]
struct ChunkDebug {
    seconds: Vec<f64>,
    similarities: Vec<serde_json::Value>,
}

pub struct AudioPatternDetector {
    clips: Vec<ClipData>,
    debug_mode: bool,
    debug_dir: PathBuf,
    height_min: f32,
    target_sample_rate: u32,
    seconds_per_chunk: u32,
    min_chunk_size: u32,
    /// Clips sharing a section, grouped by sliding window.
    section_groups: Vec<SectionGroup>,
}

impl AudioPatternDetector {
    /// Prepare the detector for a set of uniquely named clips.
    pub fn new(audio_clips: Vec<AudioClip>, options: DetectorOptions) -> Result<Self> {
        let sr = options.target_sample_rate;
        validate_sample_rate(sr)?;
        let mut debug_mode = options.debug_mode;

        let mut names = HashSet::new();
        let mut max_clip_length = 0;
        for audio_clip in &audio_clips {
            if !names.insert(audio_clip.name.as_str()) {
                return Err(Error::invalid(format!("clip {} needs to be unique", audio_clip.name)));
            }
            if audio_clip.sample_rate != sr {
                return Err(Error::invalid(format!(
                    "clip {} needs to be {sr} sample rate",
                    audio_clip.name
                )));
            }
            max_clip_length = max_clip_length.max(audio_clip.audio.len());
        }

        let seconds_per_chunk = match options.seconds_per_chunk {
            Some(seconds) if seconds >= 1 => seconds,
            _ => {
                let seconds = (max_clip_length as f64 / sr as f64).ceil() as u32 * 2;
                eprintln!(
                    "seconds_per_chunk is not set or less than 1, setting it to longest clip * 2 seconds, \
                     which is {seconds} seconds"
                );
                seconds
            }
        };

        // Validate seconds_per_chunk against all clips' sliding windows and
        // track the largest minimum chunk size across all clips.
        let mut min_chunk_size = 0;
        for audio_clip in &audio_clips {
            let clip_seconds = audio_clip.clip_length_seconds();
            let sliding_window = clip_seconds.ceil() as u32;
            let clip_min_chunk_size = sliding_window * 2;
            min_chunk_size = min_chunk_size.max(clip_min_chunk_size);
            if seconds_per_chunk < clip_min_chunk_size {
                return Err(Error::invalid(format!(
                    "seconds_per_chunk {seconds_per_chunk} is too small for clip '{}' \
                     (duration: {clip_seconds:.2}s, sliding_window: {sliding_window}s, \
                     minimum chunk size: {clip_min_chunk_size}s)",
                    audio_clip.name
                )));
            }
        }

        if debug_mode && seconds_per_chunk != DEFAULT_SECONDS_PER_CHUNK {
            eprintln!(
                "seconds_per_chunk {seconds_per_chunk} is not 60 seconds, turning off debug mode \
                 because it was made for 60 seconds only"
            );
            debug_mode = false;
        }

        let clips: Vec<ClipData> = audio_clips
            .into_iter()
            .map(|audio_clip| Self::prepare_clip(audio_clip, sr, debug_mode))
            .collect();
        let section_groups = section_groups(&clips);

        Ok(Self {
            clips,
            debug_mode,
            debug_dir: options.debug_dir,
            height_min: options.height_min.unwrap_or(DEFAULT_HEIGHT_MIN),
            target_sample_rate: sr,
            seconds_per_chunk,
            min_chunk_size,
            section_groups,
        })
    }

    /// Pre-compute everything about a clip that doesn't depend on the audio stream.
    fn prepare_clip(audio_clip: AudioClip, sr: u32, debug_mode: bool) -> ClipData {
        let clip_seconds = audio_clip.clip_length_seconds();
        let sliding_window = clip_seconds.ceil() as u32;
        if sliding_window as f64 != clip_seconds {
            eprintln!(
                "adjusted sliding_window from {clip_seconds} to {sliding_window} for {}",
                audio_clip.name
            );
        }

        let mut clip = audio_clip.audio;
        normalize_loudness(&mut clip, sr);
        let (correlation_clip, correlation_clip_absolute_max) = clip_correlation(&clip);

        if debug_mode {
            eprintln!("clip_length {} {}", audio_clip.name, clip.len());
            eprintln!("clip_length {} seconds {}", audio_clip.name, clip.len() as f64 / sr as f64);
            eprintln!("correlation_clip_length {}", correlation_clip.len());
        }

        let tone = match audio_clip.strategy {
            Some(Strategy::MarkerTone(params)) => params
                .dominant_frequency_hz
                // Fall back to deriving the frequency from the clip; if it is
                // not a pure tone the clip uses the normal verification path.
                .or_else(|| get_pure_tone_frequency(&clip, sr))
                .map(|dominant_frequency| ToneVerifier {
                    dominant_frequency,
                    thresholds: params.thresholds,
                }),
            None => None,
        };

        let is_short_clip = clip_seconds < SHORT_CLIP_DURATION_THRESHOLD;
        let pearson_windows = pearson_window_specs(is_short_clip)
            .0
            .iter()
            .map(|&window| downsample_window(&correlation_clip, window))
            .collect();

        ClipData {
            name: audio_clip.name,
            clip,
            sliding_window,
            correlation_clip,
            correlation_clip_absolute_max,
            tone,
            pearson_windows,
        }
    }

    /// Seconds per chunk actually used (after auto-computation).
    pub fn seconds_per_chunk(&self) -> u32 {
        self.seconds_per_chunk
    }

    pub fn target_sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    /// Whether debug output is active (it is turned off for non-default chunk sizes).
    pub fn debug_mode(&self) -> bool {
        self.debug_mode
    }

    /// Whether candidates for the named clip are verified as a marker tone.
    pub fn uses_marker_tone(&self, clip_name: &str) -> bool {
        self.clips.iter().any(|c| c.name == clip_name && c.tone.is_some())
    }

    /// Return the configuration values computed at construction.
    pub fn get_config(&self) -> DetectorConfig {
        let clips = self
            .clips
            .iter()
            .map(|clip_data| {
                let duration = clip_data.clip.len() as f64 / self.target_sample_rate as f64;
                let config = ClipConfig {
                    duration_seconds: (duration * 1e6).round() / 1e6,
                    sliding_window_seconds: clip_data.sliding_window,
                };
                (clip_data.name.clone(), config)
            })
            .collect();

        DetectorConfig {
            default_seconds_per_chunk: DEFAULT_SECONDS_PER_CHUNK,
            min_chunk_size_seconds: self.min_chunk_size,
            sample_rate: self.target_sample_rate,
            clips,
        }
    }

    /// Find clip occurrences in an audio stream.
    ///
    /// `on_pattern_detected` is called for each detection as soon as its
    /// chunk is processed, in timestamp order within the chunk. When
    /// `accumulate_results` is false no timestamps are collected (saves
    /// memory for long streams) and `None` is returned for them.
    ///
    /// Returns `(peak times per clip, total seconds processed)`.
    pub fn find_clip_in_audio(
        &self,
        audio_stream: &mut AudioStream,
        mut on_pattern_detected: Option<PatternDetectedCallback>,
        accumulate_results: bool,
    ) -> Result<(Option<PeakTimes>, f64)> {
        let sr = self.target_sample_rate;
        if audio_stream.sample_rate != sr {
            return Err(Error::invalid(format!(
                "full_streaming_audio_clip {} needs to be {sr} sample rate",
                audio_stream.name
            )));
        }

        let mut all_peak_times: Option<PeakTimes> = accumulate_results
            .then(|| self.clips.iter().map(|c| (c.name.clone(), Vec::new())).collect());

        let chunk_samples = self.seconds_per_chunk as usize * sr as usize;
        // Buffer to maintain continuity between chunks.
        let mut previous_chunk: Option<Vec<f32>> = None;
        let mut total_time = 0.0_f64;
        let mut index = 0usize;
        // Peaks (as absolute sample positions) each clip produced in the
        // previous chunk. The next chunk sees the end of that audio again
        // through its lookback and would report the same matches twice.
        let mut previous_peaks: Vec<Vec<i64>> = vec![Vec::new(); self.clips.len()];
        let mut buffers = RunBuffers {
            workspace: CorrelationWorkspace::new(),
            templates: self.clips.iter().map(|c| CorrelationTemplate::new(&c.clip)).collect(),
            audio_section: Vec::new(),
            correlation: Vec::new(),
        };

        loop {
            let chunk = audio_stream.source.read_samples(chunk_samples)?;
            if chunk.is_empty() {
                break;
            }
            total_time += chunk.len() as f64 / sr as f64;

            // All matches from all clips for this chunk.
            let mut chunk_matches: Vec<(f64, &str)> = Vec::new();

            for group in &self.section_groups {
                let lookback_samples = self.load_section(&chunk, previous_chunk.as_deref(), group, &mut buffers);
                for &clip_index in &group.clip_indices {
                    let clip_data = &self.clips[clip_index];
                    let seen_peaks = &mut previous_peaks[clip_index];
                    let peaks = self.process_section(clip_index, lookback_samples, index, &mut buffers)?;
                    let peak_times: Vec<f64> = peaks
                        .iter()
                        .filter(|(position, _)| !seen_peaks.contains(position))
                        .map(|&(_, timestamp)| timestamp)
                        .collect();
                    *seen_peaks = peaks.into_iter().map(|(position, _)| position).collect();

                    if on_pattern_detected.is_some() {
                        chunk_matches.extend(peak_times.iter().map(|&t| (t, clip_data.name.as_str())));
                    }
                    if let Some(all) = all_peak_times.as_mut() {
                        all.get_mut(&clip_data.name)
                            .expect("every clip has an entry")
                            .extend(peak_times);
                    }
                }
            }

            // Call the callback in timestamp order (monotonic output).
            if let Some(callback) = on_pattern_detected.as_mut() {
                chunk_matches.sort_by(|a, b| a.0.total_cmp(&b.0));
                for (timestamp, clip_name) in chunk_matches {
                    callback(clip_name, timestamp);
                }
            }

            previous_chunk = Some(chunk);
            index += 1;
        }

        Ok((all_peak_times, total_time))
    }

    /// Build the audio section of a chunk for one group of clips into
    /// `buffers`, loudness-normalize it and transform it for correlation.
    /// Returns the number of lookback samples prepended.
    ///
    /// The last `lookback_seconds` (= ceil(clip seconds)) of `previous_chunk`
    /// are prepended so a pattern that crosses the boundary is fully
    /// contained in the section. This is applied uniformly to every
    /// non-first chunk — including the final short chunk, whose own length
    /// is not a reliable lookback. The section depends only on the group's
    /// lookback, so a clip's detections never depend on other clips.
    fn load_section(
        &self,
        chunk: &[f32],
        previous_chunk: Option<&[f32]>,
        group: &SectionGroup,
        buffers: &mut RunBuffers,
    ) -> usize {
        let sr = self.target_sample_rate;
        let lookback_samples =
            previous_chunk.map_or(0, |previous| (group.lookback_seconds as usize * sr as usize).min(previous.len()));

        let RunBuffers { workspace, audio_section, .. } = buffers;
        audio_section.clear();
        if let Some(previous) = previous_chunk {
            audio_section.extend_from_slice(&previous[previous.len() - lookback_samples..]);
        }
        audio_section.extend_from_slice(chunk);

        normalize_loudness(audio_section, sr);
        // NaN comes from loudness normalization of silence.
        audio_section.iter_mut().filter(|v| v.is_nan()).for_each(|v| *v = 0.0);

        workspace.load_signal(audio_section, group.max_clip_length);
        lookback_samples
    }

    /// Detect one clip in the loaded section; returns, per match, the peak's
    /// sample position from the beginning of the stream and the timestamp of
    /// the clip start in seconds from the beginning of the stream.
    fn process_section(
        &self,
        clip_index: usize,
        lookback_samples: usize,
        index: usize,
        buffers: &mut RunBuffers,
    ) -> Result<Vec<(i64, f64)>> {
        let sr = self.target_sample_rate;
        let clip_data = &self.clips[clip_index];
        let clip_seconds = clip_data.clip.len() as f64 / sr as f64;
        let lookback_seconds = lookback_samples as f64 / sr as f64;

        let RunBuffers { workspace, templates, audio_section, correlation } = buffers;
        workspace.correlate(&mut templates[clip_index], correlation);
        let peaks = self.correlation_method(clip_data, audio_section, correlation, index)?;

        let section_start = index as i64 * self.seconds_per_chunk as i64 * sr as i64 - lookback_samples as i64;
        Ok(peaks
            .into_iter()
            .map(|peak| {
                let peak_time = peak as f64 / sr as f64 - lookback_seconds;
                let from_beginning = peak_time + (index as f64 * self.seconds_per_chunk as f64);
                // Move the timestamp to be before the clip.
                (section_start + peak as i64, (from_beginning - clip_seconds).max(0.0))
            })
            .collect())
    }

    /// Step 1 (peak finding on the cross-correlation of the section with
    /// the clip, given in `correlation`) followed by Step 2 (verification)
    /// for one audio section. Returns accepted peak indices.
    fn correlation_method(
        &self,
        clip_data: &ClipData,
        audio_section: &[f32],
        correlation: &mut [f32],
        index: usize,
    ) -> Result<Vec<usize>> {
        let clip_name = &clip_data.name;
        let clip_length = clip_data.clip.len();
        let correlation_clip_length = clip_data.correlation_clip.len();

        // Normalize by the larger of the two peaks so a much softer section
        // cannot look like a full-strength match.
        let absolute_max = absolute_in_place(correlation);
        let max_choose = clip_data.correlation_clip_absolute_max.max(absolute_max);
        correlation.iter_mut().for_each(|v| *v /= max_choose);

        let section_ts = seconds_to_time_whole(index as f64 * self.seconds_per_chunk as f64);
        if self.debug_mode {
            eprintln!("---");
            eprintln!("section_ts: {section_ts}, index {index}");
        }

        // No repetition within the duration of the clip. The height is kept
        // low so weak candidates are not missed; they are verified below.
        let peaks = find_peaks_1d(
            correlation,
            &FindPeaksOptions { height: Some(self.height_min), distance: Some(clip_length), prominence: None },
        );

        let mut peaks_final = Vec::new();
        let mut debug = ChunkDebug::default();

        for &peak in &peaks {
            // Make sure the slice is not out of bounds at the beginning and end.
            let after = peak + correlation_clip_length / 2;
            let before = peak as isize - (correlation_clip_length / 2) as isize;
            if after > correlation.len() + 5 {
                eprintln!(
                    "{section_ts} {clip_name} peak {peak} after is {after} > len(correlation)+5 {}, skipping",
                    correlation.len() + 5
                );
                continue;
            } else if before < -5 {
                eprintln!("{section_ts} {clip_name} peak {peak} before is {before} < -5, skipping");
                continue;
            }

            let accepted = match &clip_data.tone {
                Some(tone) => self.verify_marker_tone(tone, audio_section, peak, clip_length, &section_ts),
                None => self.verify_correlation_envelope(clip_data, correlation, peak, &section_ts, &mut debug),
            };
            if accepted {
                peaks_final.push(peak);
            }

            if self.debug_mode {
                self.write_debug_audio_section(clip_data, audio_section, peak, index, &section_ts)?;
            }
        }

        if self.debug_mode && !peaks.is_empty() {
            let peak_dir = self
                .debug_dir
                .join("debug")
                .join(format!("cross_correlation_{}", safe_path_component(clip_name)));
            std::fs::create_dir_all(&peak_dir)?;
            let dump = json!({
                "peaks": peaks,
                "seconds": debug.seconds,
                "similarities": debug.similarities,
            });
            let mut text = serde_json::to_string_pretty(&dump).expect("debug dump is valid JSON");
            text.push('\n');
            let file_name = format!("{index}_{}.txt", safe_path_component(&section_ts));
            std::fs::write(peak_dir.join(file_name), text)?;
            eprintln!("---");
        }

        Ok(peaks_final)
    }

    /// Save the audio around a candidate so it can be listened to.
    fn write_debug_audio_section(
        &self,
        clip_data: &ClipData,
        audio_section: &[f32],
        peak: usize,
        index: usize,
        section_ts: &str,
    ) -> Result<()> {
        let clip_name = safe_path_component(&clip_data.name);
        let section_ts = safe_path_component(section_ts);
        let audio_dir = self.debug_dir.join("audio_section").join(&clip_name);
        std::fs::create_dir_all(&audio_dir)?;

        let clip_length = clip_data.clip.len();
        let start = peak.saturating_sub(clip_length).min(audio_section.len());
        let end = (peak + clip_length).min(audio_section.len());
        write_wav_file(
            audio_dir.join(format!("{clip_name}_{index}_{section_ts}_{peak}.wav")),
            &audio_section[start..end],
            self.target_sample_rate,
        )
    }

    /// Verify a synthesized marker tone via short-time spectral analysis.
    ///
    /// The clip strategy provides a dominant frequency and optional
    /// per-station thresholds. This path targets short beeps that stay
    /// strongly single-frequency in the candidate window but can leak a
    /// little narrowband energy into one adjacent flank because of AAC
    /// smearing or the surrounding program bed.
    fn verify_marker_tone(
        &self,
        tone: &ToneVerifier,
        audio_section: &[f32],
        peak: usize,
        clip_length: usize,
        section_ts: &str,
    ) -> bool {
        let (metrics, left, right) = analyze_tone_candidate_context(
            audio_section,
            peak,
            clip_length,
            tone.dominant_frequency,
            self.target_sample_rate,
        );
        let accepted = marker_tone_accepts(tone.dominant_frequency, &tone.thresholds, &metrics, &left, &right);

        if self.debug_mode {
            if !is_close(metrics.detected_frequency, tone.dominant_frequency, 0.05, 0.0) {
                eprintln!(
                    "failed marker tone check for {section_ts}: dominant {:.1}Hz != expected {:.1}Hz",
                    metrics.detected_frequency, tone.dominant_frequency
                );
            } else {
                eprintln!(
                    "{} {section_ts}: band_purity={:.3} active_ratio={:.3} run={} active_purity={:.3} \
                     freq={:.1}Hz flank_purity=({:.3}, {:.3})",
                    if accepted { "accepted marker tone" } else { "failed marker tone check for" },
                    metrics.overall_band_purity,
                    metrics.active_frame_ratio,
                    metrics.longest_active_run,
                    metrics.active_frame_mean_purity,
                    metrics.detected_frequency,
                    left.overall_band_purity,
                    right.overall_band_purity,
                );
            }
        }
        accepted
    }

    /// Verify a candidate by comparing the correlation envelope around the
    /// peak with the clip's self-correlation: partitioned MSE first, then
    /// Pearson r on the downsampled center window.
    fn verify_correlation_envelope(
        &self,
        clip_data: &ClipData,
        correlation: &[f32],
        peak: usize,
        section_ts: &str,
        debug: &mut ChunkDebug,
    ) -> bool {
        let correlation_clip = &clip_data.correlation_clip;
        let sr = self.target_sample_rate;

        let mut correlation_slice = slicing_with_zero_padding(correlation, correlation_clip.len(), peak);
        let slice_max = correlation_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        correlation_slice.iter_mut().for_each(|v| *v /= slice_max);

        let partition_size = correlation_clip.len() / PARTITION_COUNT;
        let similarity_partitions: Vec<f32> = (0..PARTITION_COUNT)
            .map(|i| {
                let range = i * partition_size..(i + 1) * partition_size;
                mean_squared_error(&correlation_clip[range.clone()], &correlation_slice[range])
            })
            .collect();

        let similarity_middle = mean(&similarity_partitions[MIDDLE_PARTITIONS]);
        let similarity_whole = mean(&similarity_partitions);

        let is_short_clip = clip_data.clip.len() as f64 / (sr as f64) < SHORT_CLIP_DURATION_THRESHOLD;
        let similarity = if is_short_clip {
            similarity_whole
        } else {
            // f32::min would skip a NaN operand; keep NaN so it is rejected below.
            if similarity_middle < similarity_whole { similarity_middle } else { similarity_whole }
        };
        let partition_summary = json!({"whole": similarity_whole as f64, "middle": similarity_middle as f64});

        // A NaN similarity is rejected too.
        if similarity.is_nan() || similarity > SIMILARITY_HARD_LIMIT {
            if self.debug_mode {
                debug.seconds.push(peak as f64 / sr as f64);
                debug.similarities.push(json!([similarity as f64, partition_summary, null]));
                eprintln!(
                    "failed verification for {section_ts} due to similarity {similarity} > {SIMILARITY_HARD_LIMIT}"
                );
            }
            return false;
        }

        // Pearson r on the center window decides. The flanking windows are
        // computed only for debug output.
        let (windows, center_window_idx) = pearson_window_specs(is_short_clip);
        let window_r = |window_idx: usize| {
            let downsampled_slice = downsample_window(&correlation_slice, windows[window_idx]);
            pearson_correlation_1d(&clip_data.pearson_windows[window_idx], &downsampled_slice)
        };
        let pearson_r = window_r(center_window_idx);

        if self.debug_mode {
            eprintln!("similarity {similarity} pearson_r {pearson_r}");
            let mut pearson_summary = serde_json::Map::new();
            let mut best = (center_window_idx, pearson_r);
            for (window_idx, &(left, right, _)) in windows.iter().enumerate() {
                let r = if window_idx == center_window_idx { pearson_r } else { window_r(window_idx) };
                if r > best.1 || (r == best.1 && window_idx < best.0) {
                    best = (window_idx, r);
                }
                pearson_summary.insert(format!("pearson_w{left}_{right}"), json!(r));
            }
            let (best_left, best_right, _) = windows[best.0];
            let mut summary = serde_json::Map::new();
            summary.insert("pearson_r".into(), json!(pearson_r));
            summary.insert("best_window_left".into(), json!(best_left as f64));
            summary.insert("best_window_right".into(), json!(best_right as f64));
            summary.extend(pearson_summary);

            debug.seconds.push(peak as f64 / sr as f64);
            debug.similarities.push(json!([similarity as f64, partition_summary, summary]));
        }

        if pearson_r >= PEARSON_R_THRESHOLD {
            true
        } else {
            if self.debug_mode {
                eprintln!(
                    "failed verification for {section_ts} due to similarity {similarity} pearson_r {pearson_r}"
                );
            }
            false
        }
    }
}

/// Turn a clip name or timestamp into a single portable file name component
/// for debug output. Clip names can come from untrusted input (multiplexed
/// stdin), so path separators and the characters Windows rejects (such as the
/// `:` of a timestamp) are replaced, and `.`/`..` cannot be produced.
pub fn safe_path_component(name: &str) -> String {
    let mut component: String = name
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    if component.is_empty() || component.chars().all(|c| c == '.') {
        component.insert(0, '_');
    }
    component
}

/// Analyze the candidate window plus the clip-length windows on either side.
/// Returns `(matched, left flank, right flank)` metrics.
pub fn analyze_tone_candidate_context(
    audio_section: &[f32],
    peak: usize,
    clip_length: usize,
    dominant_frequency: f64,
    sample_rate: u32,
) -> (PureToneMetrics, PureToneMetrics, PureToneMetrics) {
    let length = clip_length as isize;
    let match_start = peak as isize - length + 1;
    let analyze = |start: isize| {
        let segment = extract_padded_segment(audio_section, start, clip_length);
        analyze_pure_tone_candidate(&segment, sample_rate, dominant_frequency)
    };
    (analyze(match_start), analyze(match_start - length), analyze(match_start + length))
}

/// Decide whether a candidate is the marker tone: the matched window must be
/// strongly narrowband at the expected frequency while both flanks stay quiet.
pub fn marker_tone_accepts(
    dominant_frequency: f64,
    thresholds: &MarkerToneThresholds,
    metrics: &PureToneMetrics,
    left_metrics: &PureToneMetrics,
    right_metrics: &PureToneMetrics,
) -> bool {
    let min_flank_purity = left_metrics.overall_band_purity.min(right_metrics.overall_band_purity);
    let max_flank_purity = left_metrics.overall_band_purity.max(right_metrics.overall_band_purity);

    is_close(metrics.detected_frequency, dominant_frequency, 0.05, 0.0)
        && metrics.overall_band_purity >= thresholds.minimum_band_purity.unwrap_or(0.95)
        && metrics.active_frame_ratio >= thresholds.minimum_active_frame_ratio.unwrap_or(0.80)
        && metrics.longest_active_run >= thresholds.minimum_longest_active_run.unwrap_or(9)
        && metrics.active_frame_mean_purity >= thresholds.minimum_active_frame_mean_purity.unwrap_or(0.92)
        && min_flank_purity <= thresholds.maximum_min_flank_purity.unwrap_or(0.25)
        && max_flank_purity <= thresholds.maximum_max_flank_purity.unwrap_or(0.65)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_path_component() {
        assert_eq!(safe_path_component("rthk_beep"), "rthk_beep");
        assert_eq!(safe_path_component("天空下的彩虹intro"), "天空下的彩虹intro");
        assert_eq!(safe_path_component("00:39:00"), "00_39_00");
        assert_eq!(safe_path_component("../../escape"), ".._.._escape");
        assert_eq!(safe_path_component("..\\..\\escape"), ".._.._escape");
        assert_eq!(safe_path_component("/etc/passwd"), "_etc_passwd");
        assert_eq!(safe_path_component("a<b>c\"d|e?f*g\0h"), "a_b_c_d_e_f_g_h");
        assert_eq!(safe_path_component(".."), "_..");
        assert_eq!(safe_path_component("."), "_.");
        assert_eq!(safe_path_component(""), "_");
    }

    #[test]
    fn test_zero_sample_rate_is_rejected() {
        let options = DetectorOptions { target_sample_rate: 0, ..DetectorOptions::default() };
        let err = AudioPatternDetector::new(vec![AudioClip::new("clip", vec![0.5; 8], 0)], options)
            .err()
            .expect("zero sample rate should be rejected");
        assert_eq!(err.to_string(), "target sample rate must be greater than 0");
    }

    #[test]
    fn test_slice_odd() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(slicing_with_zero_padding(&data, 5, 3), vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_slice_even() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(slicing_with_zero_padding(&data, 4, 3), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_slice_zero_pads_past_the_end() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(slicing_with_zero_padding(&data, 4, 6), vec![5.0, 6.0, 7.0, 0.0]);
        assert_eq!(slicing_with_zero_padding(&data, 5, 6), vec![5.0, 6.0, 7.0, 0.0, 0.0]);
    }

    #[test]
    fn test_slice_zero_pads_before_the_beginning() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(slicing_with_zero_padding(&data, 4, 1), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(slicing_with_zero_padding(&data, 5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mean_squared_error() {
        assert_eq!(mean_squared_error(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(mean_squared_error(&[0.0, 0.0], &[1.0, 3.0]), 5.0);
    }

    #[test]
    fn test_downsample_window_bounds_round_half_away_from_zero() {
        // 25 samples, partitions 0-5 -> [0, round(12.5) = 13).
        let curve: Vec<f32> = (0..25).map(|i| i as f32).collect();
        assert_eq!(downsample_window(&curve, (0, 5, 13)), (0..13).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(downsample_window(&curve, (5, 10, 12)), (13..25).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn test_absolute_in_place() {
        let mut values = [-3.0_f32, 1.0, -0.5, 2.0, -0.0];
        assert_eq!(absolute_in_place(&mut values), 3.0);
        assert_eq!(values, [3.0, 1.0, 0.5, 2.0, 0.0]);

        let mut negative = [-0.25_f32, -4.0, -1.0];
        assert_eq!(absolute_in_place(&mut negative), 4.0);
        assert_eq!(negative, [0.25, 4.0, 1.0]);

        assert_eq!(absolute_in_place(&mut []), f32::NEG_INFINITY);
    }

    #[test]
    fn test_clip_correlation_is_normalized_with_original_peak() {
        let clip = [0.5_f32, -1.0, 0.25, 0.75];
        let energy: f32 = clip.iter().map(|v| v * v).sum();
        let (correlation, absolute_max) = clip_correlation(&clip);
        assert_eq!(correlation.len(), 2 * clip.len() - 1);
        assert!((absolute_max - energy).abs() < 1e-5, "{absolute_max} != {energy}");
        assert_eq!(correlation[clip.len() - 1], 1.0);
        assert!(correlation.iter().all(|&v| (0.0..=1.0).contains(&v)), "{correlation:?}");
    }

    fn sine_clip(name: &str, seconds: f64) -> AudioClip {
        let sr = DEFAULT_TARGET_SAMPLE_RATE;
        let samples = (sr as f64 * seconds) as usize;
        let audio = (0..samples)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr as f64).sin() as f32)
            .collect();
        AudioClip::new(name, audio, sr)
    }

    #[test]
    fn test_section_groups_by_sliding_window_in_order_of_first_appearance() {
        let sr = DEFAULT_TARGET_SAMPLE_RATE as usize;
        let detector = AudioPatternDetector::new(
            vec![
                sine_clip("short", 0.23), // window 1
                sine_clip("long", 2.5),   // window 3
                sine_clip("medium", 0.9), // window 1
                sine_clip("longer", 3.0), // window 3 (exactly 3 seconds)
                sine_clip("one", 1.0),    // window 1 (exactly 1 second)
                sine_clip("two", 1.2),    // window 2
            ],
            DetectorOptions::default(),
        )
        .unwrap();

        let groups = &detector.section_groups;
        assert_eq!(groups.len(), 3);

        assert_eq!(groups[0].lookback_seconds, 1);
        assert_eq!(groups[0].clip_indices, vec![0, 2, 4]);
        assert_eq!(groups[0].max_clip_length, sr);

        assert_eq!(groups[1].lookback_seconds, 3);
        assert_eq!(groups[1].clip_indices, vec![1, 3]);
        assert_eq!(groups[1].max_clip_length, 3 * sr);

        assert_eq!(groups[2].lookback_seconds, 2);
        assert_eq!(groups[2].clip_indices, vec![5]);
        assert_eq!(groups[2].max_clip_length, (1.2 * sr as f64) as usize);

        // Every clip is in exactly one group.
        let mut all: Vec<usize> = groups.iter().flat_map(|g| g.clip_indices.iter().copied()).collect();
        all.sort_unstable();
        assert_eq!(all, (0..detector.clips.len()).collect::<Vec<_>>());
        for group in groups {
            for &clip_index in &group.clip_indices {
                assert_eq!(detector.clips[clip_index].sliding_window, group.lookback_seconds);
                assert!(detector.clips[clip_index].clip.len() <= group.max_clip_length);
            }
        }
    }

    #[test]
    fn test_section_groups_empty() {
        assert!(section_groups(&[]).is_empty());
    }

    #[test]
    fn test_detector_config_serializes_clips_as_ordered_map() {
        let config = DetectorConfig {
            default_seconds_per_chunk: 60,
            min_chunk_size_seconds: 4,
            sample_rate: 8000,
            clips: vec![
                ("zeta".into(), ClipConfig { duration_seconds: 1.5, sliding_window_seconds: 2 }),
                ("alpha".into(), ClipConfig { duration_seconds: 0.23, sliding_window_seconds: 1 }),
            ],
        };
        assert_eq!(
            serde_json::to_string(&config).unwrap(),
            "{\"default_seconds_per_chunk\":60,\"min_chunk_size_seconds\":4,\"sample_rate\":8000,\
             \"clips\":{\"zeta\":{\"duration_seconds\":1.5,\"sliding_window_seconds\":2},\
             \"alpha\":{\"duration_seconds\":0.23,\"sliding_window_seconds\":1}}}"
        );
    }
}
