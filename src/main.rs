use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use serde_json::{json, Map, Value};

use audio_pattern_detector::time_format::{seconds_to_ms, seconds_to_time};
use audio_pattern_detector::{
    match_pattern, match_pattern_multiplexed, match_pattern_wav_stream, AudioClip, AudioPatternDetector,
    DetectorOptions, Error, MatchOptions, Result, DEFAULT_TARGET_SAMPLE_RATE,
};

#[derive(Parser)]
#[command(name = "audio-pattern-detector", version, about = "Audio pattern detection tools")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Find pattern matches in audio files
    Match(MatchArgs),
    /// Show computed configuration for a pattern file
    ShowConfig(ShowConfigArgs),
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TimestampFormat {
    /// Integer milliseconds only
    Ms,
    /// HH:MM:SS.mmm strings only
    Formatted,
    /// Both integer milliseconds and HH:MM:SS.mmm strings
    Both,
}

#[derive(clap::Args)]
struct MatchArgs {
    /// Single audio file to find pattern in (omit when using --stdin or --multiplexed-stdin)
    audio_file: Option<PathBuf>,

    /// Pattern file, .wav or .apd.toml (can be specified multiple times)
    #[arg(long, value_name = "FILE")]
    pattern_file: Vec<PathBuf>,

    /// Folder with pattern clips (can be specified multiple times, can be combined with --pattern-file)
    #[arg(long, value_name = "DIR")]
    pattern_folder: Vec<PathBuf>,

    /// Read audio from stdin in WAV format
    #[arg(long)]
    stdin: bool,

    /// Read patterns and audio from stdin using the multiplexed protocol:
    /// [uint32 num_patterns] then for each pattern [uint32 name_len][name][uint32 data_len][wav_data],
    /// followed by the audio stream (WAV)
    #[arg(long)]
    multiplexed_stdin: bool,

    /// Target sample rate for processing in Hz
    #[arg(long, value_name = "RATE", default_value_t = DEFAULT_TARGET_SAMPLE_RATE)]
    target_sample_rate: u32,

    /// Timestamp fields in JSONL output
    #[arg(long, value_enum, default_value = "both")]
    timestamp_format: TimestampFormat,

    /// Seconds per chunk for sliding window (use "auto" to auto-compute based on pattern length)
    #[arg(long, value_name = "SECONDS", default_value = "60")]
    chunk_seconds: String,

    /// Debug mode: diagnostics on stderr plus candidate audio and peak dumps under --debug-dir
    #[arg(long)]
    debug: bool,

    /// Base directory for debug output
    #[arg(long, value_name = "DIR", default_value = "./tmp")]
    debug_dir: PathBuf,

    /// Override minimum correlation peak height (default: 0.25, lower to find weak matches)
    #[arg(long, value_name = "HEIGHT")]
    height_min: Option<f32>,
}

#[derive(clap::Args)]
struct ShowConfigArgs {
    /// Pattern file, .wav or .apd.toml
    pattern_file: PathBuf,

    /// Target sample rate for processing in Hz
    #[arg(long, value_name = "RATE", default_value_t = DEFAULT_TARGET_SAMPLE_RATE)]
    target_sample_rate: u32,
}

/// Emit a JSONL event to stdout and flush immediately.
fn emit_jsonl(event_type: &str, fields: Map<String, Value>) -> Result<()> {
    let mut event = Map::new();
    event.insert("type".into(), json!(event_type));
    event.extend(fields);
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{}", Value::Object(event))?;
    stdout.flush()?;
    Ok(())
}

/// Build the timestamp fields of an event, e.g. `timestamp_ms` / `timestamp_formatted`.
fn timestamp_fields(prefix: &str, seconds: f64, format: TimestampFormat) -> Map<String, Value> {
    let mut fields = Map::new();
    if format != TimestampFormat::Formatted {
        fields.insert(format!("{prefix}_ms"), json!(seconds_to_ms(seconds)));
    }
    if format != TimestampFormat::Ms {
        fields.insert(format!("{prefix}_formatted"), json!(seconds_to_time(seconds)));
    }
    fields
}

/// Collect `*.wav` then `*.apd.toml` files of a folder, each group sorted by name.
fn pattern_files_in_folder(folder: &Path) -> Result<Vec<PathBuf>> {
    let mut wav_files = Vec::new();
    let mut apd_files = Vec::new();
    let entries = std::fs::read_dir(folder)
        .map_err(|e| Error::invalid(format!("Failed to read pattern folder {}: {e}", folder.display())))?;
    for entry in entries {
        let path = entry?.path();
        let Some(name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
            continue;
        };
        if name.starts_with('.') || !path.is_file() {
            continue;
        }
        if name.ends_with(".apd.toml") {
            apd_files.push(path);
        } else if name.ends_with(".wav") {
            wav_files.push(path);
        }
    }
    wav_files.sort();
    apd_files.sort();
    wav_files.extend(apd_files);
    Ok(wav_files)
}

fn cmd_match(args: MatchArgs) -> Result<()> {
    // "auto" (or anything below 1) auto-computes the chunk size.
    let seconds_per_chunk = if args.chunk_seconds.eq_ignore_ascii_case("auto") {
        None
    } else {
        let seconds: i64 = args.chunk_seconds.parse().map_err(|_| {
            Error::invalid(format!(
                "--chunk-seconds must be 'auto' or a positive integer, got '{}'",
                args.chunk_seconds
            ))
        })?;
        u32::try_from(seconds).ok().filter(|&s| s >= 1)
    };

    let options = MatchOptions {
        debug_mode: args.debug,
        seconds_per_chunk,
        target_sample_rate: args.target_sample_rate,
        debug_dir: args.debug_dir,
        height_min: args.height_min,
    };

    let mut pattern_files: Vec<PathBuf> = Vec::new();
    if !args.multiplexed_stdin {
        for folder in &args.pattern_folder {
            for pattern_file in pattern_files_in_folder(folder)? {
                eprintln!("adding pattern file {}...", pattern_file.display());
                pattern_files.push(pattern_file);
            }
        }
        pattern_files.extend(args.pattern_file);
        if pattern_files.is_empty() {
            return Err(Error::invalid(
                "Please provide either --pattern-file, --pattern-folder, or --multiplexed-stdin",
            ));
        }
        if !args.stdin && args.audio_file.is_none() {
            return Err(Error::invalid("Please provide an audio file or --stdin or --multiplexed-stdin"));
        }
    }

    let timestamp_format = args.timestamp_format;
    let mut emit_error: Option<Error> = None;
    let mut last_ms: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let mut callback = |clip_name: &str, timestamp: f64| {
        // Overlapping sections can report the same detection twice.
        let ts_ms = seconds_to_ms(timestamp);
        if last_ms.get(clip_name) == Some(&ts_ms) || emit_error.is_some() {
            return;
        }
        last_ms.insert(clip_name.to_string(), ts_ms);

        let mut fields = Map::new();
        fields.insert("clip_name".into(), json!(clip_name));
        fields.extend(timestamp_fields("timestamp", timestamp, timestamp_format));
        if let Err(e) = emit_jsonl("pattern_detected", fields) {
            emit_error = Some(e);
        }
    };

    let source = if args.multiplexed_stdin {
        "multiplexed-stdin".to_string()
    } else if args.stdin {
        "stdin".to_string()
    } else {
        args.audio_file.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
    };
    let mut start_fields = Map::new();
    start_fields.insert("source".into(), json!(source));
    emit_jsonl("start", start_fields)?;

    let (_, total_time) = if args.multiplexed_stdin {
        match_pattern_multiplexed(std::io::stdin().lock(), &options, Some(&mut callback), false)?
    } else if args.stdin {
        match_pattern_wav_stream(std::io::stdin().lock(), &pattern_files, &options, Some(&mut callback), false)?
    } else {
        let audio_file = args.audio_file.expect("audio file presence checked above");
        match_pattern(audio_file, &pattern_files, &options, Some(&mut callback), false)?
    };
    if let Some(e) = emit_error {
        return Err(e);
    }

    eprintln!("Total time processed: {}", seconds_to_time(total_time));
    emit_jsonl("end", timestamp_fields("total_time", total_time, timestamp_format))
}

fn cmd_show_config(args: ShowConfigArgs) -> Result<()> {
    if !args.pattern_file.exists() {
        return Err(Error::invalid(format!("Pattern {} does not exist", args.pattern_file.display())));
    }
    let pattern_clip = AudioClip::from_audio_file(&args.pattern_file, args.target_sample_rate)?;

    // Auto mode shows the minimum computed values.
    let detector = AudioPatternDetector::new(
        vec![pattern_clip],
        DetectorOptions {
            seconds_per_chunk: None,
            target_sample_rate: args.target_sample_rate,
            ..DetectorOptions::default()
        },
    )?;
    let config = serde_json::to_string_pretty(&detector.get_config()).expect("config is valid JSON");
    println!("{config}");
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Command::Match(args)) => cmd_match(args),
        Some(Command::ShowConfig(args)) => cmd_show_config(args),
        None => {
            let _ = Cli::command().print_help();
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
