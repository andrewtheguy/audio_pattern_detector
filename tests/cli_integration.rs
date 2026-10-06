//! CLI integration tests for audio-pattern-detector.
//!
//! Tests the command-line interface to verify:
//! 1. Arguments are parsed and passed correctly to internal functions
//! 2. Output format is correct (JSONL events, JSON config)
//! 3. Error handling for CLI-specific errors
//!
//! Internal detection logic is tested in the other integration test files.

use std::collections::BTreeSet;
use std::io::Write;
use std::process::{Command, Output, Stdio};

use audio_pattern_detector::stream::resample_audio;
use audio_pattern_detector::wav::{encode_wav_i16, load_wav_file};
use serde_json::Value;

const CLI_BIN: &str = env!("CARGO_BIN_EXE_audio-pattern-detector");

const RTHK_BEEP_PATTERN: &str = "sample_audios/clips/rthk_beep.apd.toml";
const RTHK_BEEP_AUDIO: &str = "sample_audios/rthk_section_with_beep.wav";
const RTHK_BEEP_AUDIO_NAME: &str = "rthk_section_with_beep.wav";
const RTHK_BEEP_AUDIO_16K: &str = "sample_audios/test_16khz/rthk_section_with_beep_16k.wav";
const RTHK_BEEP_CLIP_NAME: &str = "rthk_beep";
const CBS_NEWS_PATTERN: &str = "sample_audios/clips/cbs_news.wav";
const CBS_NEWS_AUDIO: &str = "sample_audios/cbs_news_audio_section.wav";
const CBS_NEWS_CLIP_NAME: &str = "cbs_news";
const RAINBOW_INTRO_PATTERN: &str = "sample_audios/clips/天空下的彩虹intro.wav";
const PATTERN_FOLDER: &str = "sample_audios/clips";
const NONEXISTENT_FILE: &str = "nonexistent.wav";
const DEFAULT_SAMPLE_RATE: u32 = 8000;

struct CliResult {
    code: i32,
    stdout: String,
    stderr: String,
}

impl CliResult {
    fn from_output(output: Output) -> Self {
        Self {
            code: output.status.code().expect("CLI terminated by a signal"),
            stdout: String::from_utf8(output.stdout).expect("stdout is UTF-8"),
            stderr: String::from_utf8(output.stderr).expect("stderr is UTF-8"),
        }
    }

    /// Parse stdout as JSONL, one event per line.
    fn events(&self) -> Vec<Value> {
        self.stdout
            .trim()
            .split('\n')
            .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("invalid JSONL line {line:?}: {e}")))
            .collect()
    }
}

/// Run the CLI without asserting on the exit code (Python's `check=False`).
fn run_cli_unchecked(args: &[&str]) -> CliResult {
    let output = Command::new(CLI_BIN)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("failed to run CLI");
    CliResult::from_output(output)
}

/// Run the CLI and require a zero exit code (Python's `check=True`).
fn run_cli(args: &[&str]) -> CliResult {
    let result = run_cli_unchecked(args);
    assert_eq!(result.code, 0, "CLI {args:?} failed: {}", result.stderr);
    result
}

/// Run the CLI with binary stdin data, without asserting on the exit code.
fn run_cli_stdin_unchecked(args: &[&str], stdin_data: Vec<u8>) -> CliResult {
    let mut child = Command::new(CLI_BIN)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn CLI");
    let mut stdin = child.stdin.take().expect("child stdin is piped");
    // Write from a separate thread so a large payload cannot deadlock against the output pipes.
    // The CLI may exit before consuming everything (e.g. on a rejected header), so a broken
    // pipe is not an error here.
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&stdin_data);
    });
    let output = child.wait_with_output().expect("failed to wait for CLI");
    writer.join().expect("stdin writer thread panicked");
    CliResult::from_output(output)
}

/// Run the CLI with binary stdin data and require a zero exit code.
fn run_cli_stdin(args: &[&str], stdin_data: Vec<u8>) -> CliResult {
    let result = run_cli_stdin_unchecked(args, stdin_data);
    assert_eq!(result.code, 0, "CLI {args:?} failed: {}", result.stderr);
    result
}

/// Convert an audio file to mono 16-bit PCM WAV bytes at `sample_rate` (Python piped it through ffmpeg).
fn convert_file_to_wav_bytes(audio_file: &str, sample_rate: u32) -> Vec<u8> {
    let (audio, orig_sr) = load_wav_file(audio_file).expect("failed to load audio file");
    let audio = resample_audio(audio, orig_sr, sample_rate);
    encode_wav_i16(&audio, sample_rate)
}

/// Build a multiplexed stdin payload.
///
/// Protocol:
///     [4 bytes] number_of_patterns (uint32 little-endian)
///     For each pattern:
///         [4 bytes] name_length (uint32 little-endian)
///         [name_length bytes] name (UTF-8)
///         [4 bytes] data_length (uint32 little-endian)
///         [data_length bytes] WAV data
///     [remaining bytes] audio stream
fn build_multiplexed_payload(patterns: &[(&str, &[u8])], audio_data: &[u8]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&u32::try_from(patterns.len()).unwrap().to_le_bytes());
    for (name, wav_data) in patterns {
        let name_bytes = name.as_bytes();
        payload.extend_from_slice(&u32::try_from(name_bytes.len()).unwrap().to_le_bytes());
        payload.extend_from_slice(name_bytes);
        payload.extend_from_slice(&u32::try_from(wav_data.len()).unwrap().to_le_bytes());
        payload.extend_from_slice(wav_data);
    }
    payload.extend_from_slice(audio_data);
    payload
}

fn pattern_events(events: &[Value]) -> Vec<&Value> {
    events.iter().filter(|e| e["type"] == "pattern_detected").collect()
}

fn detected_clip_names(events: &[Value]) -> BTreeSet<String> {
    pattern_events(events)
        .iter()
        .map(|e| e["clip_name"].as_str().expect("clip_name is a string").to_string())
        .collect()
}

fn has_key(event: &Value, key: &str) -> bool {
    event.as_object().expect("event is a JSON object").contains_key(key)
}

/// The key is present and holds a JSON integer (Python `isinstance(value, int)`).
fn has_int(event: &Value, key: &str) -> bool {
    event.get(key).is_some_and(|v| v.is_i64() || v.is_u64())
}

/// The key is present and holds a JSON string (Python `isinstance(value, str)`).
fn has_str(event: &Value, key: &str) -> bool {
    event.get(key).is_some_and(Value::is_string)
}

// --- Help and Basic CLI Tests ---

#[test]
fn cli_help() {
    let result = run_cli(&["--help"]);
    assert_eq!(result.code, 0);
    assert!(result.stdout.contains("audio-pattern-detector"));
    assert!(result.stdout.contains("match"));
    assert!(result.stdout.contains("show-config"));
}

#[test]
fn cli_match_help() {
    let result = run_cli(&["match", "--help"]);
    assert_eq!(result.code, 0);
    assert!(result.stdout.contains("--pattern-file"));
    assert!(result.stdout.contains("--pattern-folder"));
    assert!(result.stdout.contains("--stdin"));
    assert!(result.stdout.contains("--target-sample-rate"));
    assert!(result.stdout.contains("--chunk-seconds"));
}

#[test]
fn cli_show_config_help() {
    let result = run_cli(&["show-config", "--help"]);
    assert_eq!(result.code, 0);
    assert!(result.stdout.contains("pattern file"));
}

#[test]
fn cli_no_command() {
    let result = run_cli_unchecked(&[]);
    assert_eq!(result.code, 1);
}

// --- Match Command: Argument Passing Tests ---

#[test]
fn match_audio_file_returns_jsonl() {
    let result = run_cli(&["match", RTHK_BEEP_AUDIO, "--pattern-file", RTHK_BEEP_PATTERN]);
    assert_eq!(result.code, 0);

    let events = result.events();
    assert_eq!(events[0]["type"], "start");
    assert_eq!(events[events.len() - 1]["type"], "end");

    let pattern_events = pattern_events(&events);
    assert!(!pattern_events.is_empty());
    assert_eq!(pattern_events[0]["clip_name"], RTHK_BEEP_CLIP_NAME);
}

#[test]
fn match_pattern_folder_passes_multiple_patterns() {
    let result = run_cli(&["match", CBS_NEWS_AUDIO, "--pattern-folder", PATTERN_FOLDER]);
    assert_eq!(result.code, 0);

    let clip_names = detected_clip_names(&result.events());
    assert!(clip_names.contains(CBS_NEWS_CLIP_NAME));
}

#[test]
fn match_chunk_seconds_argument_passed() {
    // Auto mode should work without error
    let result = run_cli(&[
        "match",
        RTHK_BEEP_AUDIO,
        "--pattern-file",
        RTHK_BEEP_PATTERN,
        "--chunk-seconds",
        "auto",
    ]);
    assert_eq!(result.code, 0);

    // Explicit value
    let result = run_cli(&[
        "match",
        RTHK_BEEP_AUDIO,
        "--pattern-file",
        RTHK_BEEP_PATTERN,
        "--chunk-seconds",
        "10",
    ]);
    assert_eq!(result.code, 0);
}

#[test]
fn match_chunk_seconds_invalid_value() {
    let result = run_cli_unchecked(&[
        "match",
        RTHK_BEEP_AUDIO,
        "--pattern-file",
        RTHK_BEEP_PATTERN,
        "--chunk-seconds",
        "invalid",
    ]);
    assert_ne!(result.code, 0);
    assert!(result.stderr.contains("auto") || result.stderr.contains("integer"));
}

// --- Match Command: --stdin Tests (WAV, Always JSONL) ---

#[test]
fn match_stdin_reads_wav() {
    let wav_data = convert_file_to_wav_bytes(RTHK_BEEP_AUDIO, DEFAULT_SAMPLE_RATE);

    let result = run_cli_stdin(&["match", "--stdin", "--pattern-file", RTHK_BEEP_PATTERN], wav_data);
    assert_eq!(result.code, 0);

    let events = result.events();
    let end_event = &events[events.len() - 1];
    assert_eq!(events[0]["type"], "start");
    assert_eq!(end_event["type"], "end");

    let pattern_events = pattern_events(&events);
    assert!(!pattern_events.is_empty());
    assert_eq!(pattern_events[0]["clip_name"], RTHK_BEEP_CLIP_NAME);
    assert!(has_int(pattern_events[0], "timestamp_ms"));
    assert!(has_str(pattern_events[0], "timestamp_formatted"));
    assert!(has_int(end_event, "total_time_ms"));
    assert!(has_str(end_event, "total_time_formatted"));
}

#[test]
fn match_stdin_with_pattern_folder() {
    let wav_data = convert_file_to_wav_bytes(CBS_NEWS_AUDIO, DEFAULT_SAMPLE_RATE);

    let result = run_cli_stdin(&["match", "--stdin", "--pattern-folder", PATTERN_FOLDER], wav_data);
    assert_eq!(result.code, 0);

    // CBS news patterns should be detected in CBS audio
    let pattern_names = detected_clip_names(&result.events());
    assert!(pattern_names.contains(CBS_NEWS_CLIP_NAME));
}

// --- Match Command: JSONL Output Format Tests ---

#[test]
fn match_jsonl_output_format() {
    let result = run_cli(&["match", RTHK_BEEP_AUDIO, "--pattern-file", RTHK_BEEP_PATTERN]);
    assert_eq!(result.code, 0);

    let events = result.events();
    let end_event = &events[events.len() - 1];

    assert_eq!(events[0]["type"], "start");
    assert!(has_key(&events[0], "source"));

    // Both timestamp formats by default
    assert_eq!(end_event["type"], "end");
    assert!(has_int(end_event, "total_time_ms"));
    assert!(has_str(end_event, "total_time_formatted"));

    for event in &events[1..events.len() - 1] {
        assert_eq!(event["type"], "pattern_detected");
        assert!(has_key(event, "clip_name"));
        assert!(has_int(event, "timestamp_ms"));
        assert!(has_str(event, "timestamp_formatted"));
    }
}

#[test]
fn match_jsonl_timestamp_format_ms() {
    let result = run_cli(&[
        "match",
        RTHK_BEEP_AUDIO,
        "--pattern-file",
        RTHK_BEEP_PATTERN,
        "--timestamp-format",
        "ms",
    ]);
    assert_eq!(result.code, 0);

    let events = result.events();
    let end_event = &events[events.len() - 1];

    assert!(has_int(end_event, "total_time_ms"));
    assert!(!has_key(end_event, "total_time_formatted"));

    for event in &events[1..events.len() - 1] {
        assert_eq!(event["type"], "pattern_detected");
        assert!(has_int(event, "timestamp_ms"));
        assert!(!has_key(event, "timestamp_formatted"));
    }
}

#[test]
fn match_jsonl_timestamp_format_formatted() {
    let result = run_cli(&[
        "match",
        RTHK_BEEP_AUDIO,
        "--pattern-file",
        RTHK_BEEP_PATTERN,
        "--timestamp-format",
        "formatted",
    ]);
    assert_eq!(result.code, 0);

    let events = result.events();
    let end_event = &events[events.len() - 1];

    assert!(has_str(end_event, "total_time_formatted"));
    assert!(!has_key(end_event, "total_time_ms"));

    for event in &events[1..events.len() - 1] {
        assert_eq!(event["type"], "pattern_detected");
        assert!(has_str(event, "timestamp_formatted"));
        assert!(!has_key(event, "timestamp_ms"));
    }
}

#[test]
fn match_jsonl_start_event_source() {
    // Audio file
    let result = run_cli(&["match", RTHK_BEEP_AUDIO, "--pattern-file", RTHK_BEEP_PATTERN]);
    let start_event = &result.events()[0];
    assert!(start_event["source"].as_str().expect("source is a string").contains(RTHK_BEEP_AUDIO_NAME));

    // Stdin (WAV mode)
    let wav_data = convert_file_to_wav_bytes(RTHK_BEEP_AUDIO, DEFAULT_SAMPLE_RATE);
    let result = run_cli_stdin(&["match", "--stdin", "--pattern-file", RTHK_BEEP_PATTERN], wav_data);
    let start_event = &result.events()[0];
    assert_eq!(start_event["source"], "stdin");
}

#[test]
fn match_jsonl_no_match_only_start_end() {
    let result = run_cli(&["match", RTHK_BEEP_AUDIO, "--pattern-file", CBS_NEWS_PATTERN]);
    assert_eq!(result.code, 0);

    // Only start and end events (no pattern_detected)
    let events = result.events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["type"], "start");
    assert_eq!(events[1]["type"], "end");
}

// --- Show-config Command Tests ---

#[test]
fn show_config_returns_json() {
    let result = run_cli(&["show-config", RTHK_BEEP_PATTERN]);
    assert_eq!(result.code, 0);

    let config: Value = serde_json::from_str(&result.stdout).expect("show-config prints JSON");
    assert!(has_key(&config, "default_seconds_per_chunk"));
    assert!(has_key(&config, "min_chunk_size_seconds"));
    assert!(has_key(&config, "sample_rate"));
    assert!(has_key(&config, "clips"));
    assert!(has_key(&config["clips"], RTHK_BEEP_CLIP_NAME));
}

#[test]
fn show_config_clip_info() {
    let result = run_cli(&["show-config", RTHK_BEEP_PATTERN]);
    let config: Value = serde_json::from_str(&result.stdout).expect("show-config prints JSON");

    let clip_config = &config["clips"][RTHK_BEEP_CLIP_NAME];
    assert!(has_key(clip_config, "duration_seconds"));
    assert!(has_key(clip_config, "sliding_window_seconds"));
}

// --- Error Handling Tests ---

#[test]
fn match_nonexistent_audio_file() {
    let result = run_cli_unchecked(&["match", NONEXISTENT_FILE, "--pattern-file", RTHK_BEEP_PATTERN]);
    assert_ne!(result.code, 0);
}

#[test]
fn match_nonexistent_pattern_file() {
    let result = run_cli_unchecked(&["match", RTHK_BEEP_AUDIO, "--pattern-file", NONEXISTENT_FILE]);
    assert_ne!(result.code, 0);
}

#[test]
fn match_no_audio_source() {
    let result = run_cli_unchecked(&["match", "--pattern-file", RTHK_BEEP_PATTERN]);
    assert_ne!(result.code, 0);
    assert!(result.stderr.contains("Please provide"));
}

#[test]
fn match_no_pattern() {
    let result = run_cli_unchecked(&["match", RTHK_BEEP_AUDIO]);
    assert_ne!(result.code, 0);
    assert!(result.stderr.contains("Please provide"));
}

// There is no `convert` subcommand (in Python either): this only checks the invocation fails.
#[test]
fn convert_nonexistent_file() {
    let output_file = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .expect("failed to create temp file");
    let output_path = output_file.path().to_str().expect("temp path is UTF-8");

    let result = run_cli_unchecked(&["convert", "--audio-file", NONEXISTENT_FILE, "--dest-file", output_path]);
    assert_ne!(result.code, 0);
}

#[test]
fn show_config_no_pattern() {
    let result = run_cli_unchecked(&["show-config"]);
    assert_ne!(result.code, 0);
}

#[test]
fn show_config_nonexistent_pattern() {
    let result = run_cli_unchecked(&["show-config", NONEXISTENT_FILE]);
    assert_ne!(result.code, 0);
}

// --- 16kHz Audio Auto-Conversion Tests ---

#[test]
fn match_16khz_audio_auto_converts() {
    let result = run_cli(&["match", RTHK_BEEP_AUDIO_16K, "--pattern-file", RTHK_BEEP_PATTERN]);
    assert_eq!(result.code, 0);

    // Pattern is detected despite the sample rate difference
    let events = result.events();
    let pattern_events = pattern_events(&events);
    assert!(!pattern_events.is_empty());
    assert_eq!(pattern_events[0]["clip_name"], RTHK_BEEP_CLIP_NAME);
}

// --- Stdin with Sample Rate Tests ---

#[test]
fn stdin_wav_with_wrong_sample_rate_rejected() {
    // WAV at 16kHz (default target is 8kHz)
    let wav_data = convert_file_to_wav_bytes(RTHK_BEEP_AUDIO, 16000);

    let result = run_cli_stdin_unchecked(&["match", "--stdin", "--pattern-file", RTHK_BEEP_PATTERN], wav_data);
    assert_ne!(result.code, 0);
    assert!(result.stderr.contains("Expected 8000 Hz"), "stderr: {}", result.stderr);
}

// --- Match Command: --multiplexed-stdin Tests ---

#[test]
fn multiplexed_stdin_help() {
    let result = run_cli(&["match", "--help"]);
    assert!(result.stdout.contains("--multiplexed-stdin"));
}

#[test]
fn multiplexed_stdin_single_pattern_wav_audio() {
    // The multiplexed-stdin protocol is WAV-only, so this uses the cbs_news
    // .wav pattern (not the .apd.toml pure-tone pattern).
    let pattern_data = std::fs::read(CBS_NEWS_PATTERN).expect("failed to read pattern");
    let audio_data = convert_file_to_wav_bytes(CBS_NEWS_AUDIO, DEFAULT_SAMPLE_RATE);

    let payload = build_multiplexed_payload(&[(CBS_NEWS_CLIP_NAME, &pattern_data)], &audio_data);

    let result = run_cli_stdin(&["match", "--multiplexed-stdin"], payload);
    assert_eq!(result.code, 0);

    let events = result.events();
    assert_eq!(events[0]["type"], "start");
    assert_eq!(events[0]["source"], "multiplexed-stdin");
    assert_eq!(events[events.len() - 1]["type"], "end");

    let pattern_events = pattern_events(&events);
    assert!(!pattern_events.is_empty());
    assert_eq!(pattern_events[0]["clip_name"], CBS_NEWS_CLIP_NAME);
}

#[test]
fn multiplexed_stdin_multiple_patterns() {
    // The multiplexed-stdin protocol is WAV-only, so both patterns are .wav.
    let pattern1_data = std::fs::read(CBS_NEWS_PATTERN).expect("failed to read pattern");
    let pattern2_data = std::fs::read(RAINBOW_INTRO_PATTERN).expect("failed to read pattern");

    // Match against CBS audio: should detect cbs_news, not the rainbow intro.
    let audio_data = convert_file_to_wav_bytes(CBS_NEWS_AUDIO, DEFAULT_SAMPLE_RATE);

    let payload = build_multiplexed_payload(
        &[(CBS_NEWS_CLIP_NAME, &pattern1_data), ("rainbow_intro", &pattern2_data)],
        &audio_data,
    );

    let result = run_cli_stdin(&["match", "--multiplexed-stdin"], payload);
    assert_eq!(result.code, 0);

    let clip_names = detected_clip_names(&result.events());
    assert!(clip_names.contains(CBS_NEWS_CLIP_NAME));
}

#[test]
fn multiplexed_stdin_requires_no_pattern_file() {
    let pattern_data = std::fs::read(CBS_NEWS_PATTERN).expect("failed to read pattern");
    let audio_data = convert_file_to_wav_bytes(CBS_NEWS_AUDIO, DEFAULT_SAMPLE_RATE);

    let payload = build_multiplexed_payload(&[("test_pattern", &pattern_data)], &audio_data);

    // Works without --pattern-file or --pattern-folder
    let result = run_cli_stdin(&["match", "--multiplexed-stdin"], payload);
    assert_eq!(result.code, 0);
}
