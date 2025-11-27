# Audio Pattern Detector
This is a project that detects audio patterns specified by audio clips on a target audio file.
It is designed to detect audio patterns like intros, breaks, and outros from prerecorded radio shows and podcasts,
which can be helpful to get the sections of the particular radio show or podcast that you are interested in.

This library is particularly useful for AI workflows to efficiently segment audio files before processing. For example, it helps with OpenAI Whisper transcription preprocessing by trimming audio to the relevant sections, reducing processing time and costs.

The default sample rate of 8kHz was chosen for faster runtime and lower memory usage while not having any noticeable effects on matching results compared to the tests I have done with higher sample rates, since precise timestamps are not required for segmentation tasks. For AI workflows that require 16kHz audio (common for speech recognition models), the library can work with 16kHz as well.

It uses cross-correlation to detect the potential matching pattern, and then uses mean square error and overlapping areas on cross correlation graph to eliminate false positives.


## Installation

### Install from GitHub (recommended)
```shell
uv tool install git+https://github.com/andrewtheguy/audio_pattern_detector.git
```

### Install locally for development
```shell
uv pip install -e .
```

### Run without installing

**Using uv (from local directory):**
```shell
uv run audio-pattern-detector [command] [options]
```

**Using pipx (from GitHub):**
```shell
pipx run --spec git+https://github.com/andrewtheguy/audio_pattern_detector.git audio-pattern-detector [command] [options]
```

## Audio Requirements

- **Mono Only**: Only mono (single channel) audio is supported. Stereo or multi-channel audio must be converted to mono first.
- **Sample Rate**: All audio clips (patterns) and audio streams must use the same sample rate for matching to work. The default is 8kHz.
- **Format**: Audio files should be mono, 16-bit WAV format.
- **Converting**: Use the `convert` command to convert audio files to the required format (8kHz, mono, 16-bit WAV).

## Usage

The package provides a single CLI command `audio-pattern-detector` with two subcommands:

### Match - Detect patterns in audio files

The source audio file is automatically converted to the required format (mono, 16-bit WAV at the target sample rate) using ffmpeg.

```shell
# detect pattern from audio file, add --debug to enable debug mode
audio-pattern-detector match --audio-file ./sample_audios/cbs_news_audio_section.wav --pattern-file ./sample_audios/clips/cbs_news_dada.wav

# detect pattern using a folder of pattern clips
audio-pattern-detector match --audio-folder ./audio_files --pattern-folder ./sample_audios/clips

# with uv run (no install needed)
uv run audio-pattern-detector match --audio-file ./sample_audios/cbs_news_audio_section.wav --pattern-file ./sample_audios/clips/cbs_news_dada.wav

# with pipx
pipx run --spec . audio-pattern-detector match --audio-file ./sample_audios/cbs_news_audio_section.wav --pattern-file ./sample_audios/clips/cbs_news_dada.wav
```

#### Match CLI Options

| Option | Description |
|--------|-------------|
| `--audio-file` | Audio file to search for patterns |
| `--audio-folder` | Folder of audio files to process |
| `--audio-url` | URL to audio file (must not be a live stream) |
| `--stdin` | Read audio from stdin pipe |
| `--input-format` | Input format hint for stdin (e.g., mp3, wav, flac) |
| `--pattern-file` | Single pattern file to match |
| `--pattern-folder` | Folder of pattern clips to match |
| `--chunk-seconds` | Seconds per chunk for sliding window (default: 60, use "auto" to auto-compute based on pattern length) |
| `--show-config` | Output computed configuration as JSON and exit (no audio file required) |
| `--jsonl` | Output JSONL events as they occur (streaming mode) |
| `--debug` | Enable debug mode |

#### Sliding Window Configuration

The `--chunk-seconds` option controls the sliding window size for processing audio:

```shell
# Use default 60-second chunks
audio-pattern-detector match --audio-file audio.wav --pattern-file pattern.wav

# Auto-compute chunk size based on pattern length (2x longest pattern)
audio-pattern-detector match --audio-file audio.wav --pattern-file pattern.wav --chunk-seconds auto

# Use custom 10-second chunks
audio-pattern-detector match --audio-file audio.wav --pattern-file pattern.wav --chunk-seconds 10
```

#### Show Configuration

Use `--show-config` to see computed configuration without processing audio:

```shell
audio-pattern-detector match --pattern-folder ./clips --chunk-seconds auto --show-config
```

Output:
```json
{
  "seconds_per_chunk": 2,
  "chunk_size_bytes": 64000,
  "sample_rate": 8000,
  "min_chunk_size_seconds": 2,
  "clips": {
    "pattern1": {
      "duration_seconds": 0.5,
      "sliding_window_seconds": 1,
      "is_pure_tone": false
    }
  }
}
```

#### JSONL Streaming Output

Use `--jsonl` for streaming output that emits events as patterns are detected:

```shell
audio-pattern-detector match --audio-file audio.wav --pattern-file pattern.wav --jsonl
```

Output format:
```jsonl
{"type": "start", "source": "audio.wav"}
{"type": "pattern_detected", "clip_name": "pattern", "timestamp": 5.5, "timestamp_formatted": "00:00:05.500"}
{"type": "end", "total_time": 60.0, "total_time_formatted": "00:01:00.000"}
```

### Convert - Convert audio files to clip format
```shell
# convert audio file to target sample rate (8kHz, mono)
audio-pattern-detector convert --audio-file ./tmp/cbs_news_dada.wav --dest-file ./sample_audios/clips/cbs_news_dada.wav

# with uv run (no install needed)
uv run audio-pattern-detector convert --audio-file ./tmp/cbs_news_dada.wav --dest-file ./sample_audios/clips/cbs_news_dada.wav

# with pipx
pipx run --spec . audio-pattern-detector convert --audio-file ./tmp/cbs_news_dada.wav --dest-file ./sample_audios/clips/cbs_news_dada.wav
```

## audio pattern detection methods
currently only supports cross-correlation

### default cross-correlation
Picks all peaks that are above a certain threshold, and then eliminate false positives with cross similarity and mean square error.
Works well with repeating or non-repeating patterns that are loud enough within the audio section because it adds the normalized clip
at the end of the audio section, which helps to eliminate false positives that are much softer or non-related to the clip.
won't work well for patterns that are too short, currently it disallow short clips unless it is pure tone pattern, if it is short and pure tone pattern, then a special correlation logic is used to match. 

It will miss distorted patterns like this because error score is too high and area overlap ratio is too low:
![rthk_beep_39_00:39:00_478782](https://github.com/user-attachments/assets/80669708-b8f9-461c-ae6c-2edddb161904)

## testing
use pytest to test because not all of them are written using default python unittest module, and pytest is more flexible and easier to use.
