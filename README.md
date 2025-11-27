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

| Option | Description | Requires ffmpeg |
|--------|-------------|-----------------|
| `--audio-file` | Audio file to search for patterns | Yes |
| `--audio-folder` | Folder of audio files to process | Yes |
| `--stdin` | Read WAV format audio from stdin (always outputs JSONL) | No |
| `--raw-pcm` | With `--stdin`: read raw float32 little-endian PCM instead of WAV (requires `--source-sample-rate`) | No |
| `--source-sample-rate` | Source sample rate for raw PCM stdin in Hz (only used with `--raw-pcm`) | No |
| `--target-sample-rate` | Target sample rate for processing in Hz (default: 8000). Use 16000 for AI workflows that require 16kHz audio. | - |
| `--pattern-file` | Single pattern file to match | Yes* |
| `--pattern-folder` | Folder of pattern clips to match | Yes* |
| `--chunk-seconds` | Seconds per chunk for sliding window (default: 60, use "auto" to auto-compute based on pattern length) | - |
| `--jsonl` | Output JSONL events as they occur (streaming mode, for file inputs) | - |
| `--debug` | Enable debug mode | - |

*Pattern files require ffmpeg for automatic resampling to the target sample rate. If patterns are already at the target sample rate (default 8kHz), scipy is used as fallback.

#### Target Sample Rate

The `--target-sample-rate` option allows using a different sample rate for processing:

```shell
# Use default 8kHz sample rate (faster, lower memory)
audio-pattern-detector match --audio-file audio.wav --pattern-file pattern.wav

# Use 16kHz for AI workflows (helpful if pattern.wav is already at 16kHz)
audio-pattern-detector match --audio-file audio.wav --pattern-file pattern.wav --target-sample-rate 16000
```

**Note**: When using a custom target sample rate:
- Pattern files are automatically resampled to the target rate
- Audio files are converted to the target rate via ffmpeg
- For WAV stdin mode, audio is automatically resampled from the WAV header sample rate
- For raw PCM stdin mode, use `--source-sample-rate` for input rate and `--target-sample-rate` for processing rate

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

#### Stdin Mode (WAV)

Use `--stdin` to read WAV format audio from stdin. This mode always outputs JSONL for real-time streaming detection. The sample rate is automatically read from the WAV header.

**This mode does not require ffmpeg** - audio is read using Python's `wave` module and resampled using scipy.

```shell
# WAV stdin - sample rate read from header, resampled to 8kHz (default target) if different
ffmpeg -i input.mp3 -f wav -ac 1 pipe: | \
  audio-pattern-detector match --stdin --pattern-file pattern.wav

# WAV stdin - sample rate read from header, resampled to 16kHz target if different
ffmpeg -i input.mp3 -f wav -ac 1 pipe: | \
  audio-pattern-detector match --stdin --target-sample-rate 16000 --pattern-file pattern.wav

# WAV stdin at 8kHz - no resampling needed (source matches default target)
ffmpeg -i input.mp3 -f wav -ac 1 -ar 8000 pipe: | \
  audio-pattern-detector match --stdin --pattern-file pattern.wav
```

**Note**: When using `--stdin` (WAV mode):
- Input must be WAV format (sample rate, channels, and bit depth are read from header)
- Stereo audio is automatically mixed to mono
- Output is always JSONL format for real-time streaming
- **Resampling**: If the WAV header sample rate differs from `--target-sample-rate` (default: 8000), audio is resampled using scipy

#### Stdin Mode (Raw PCM)

Use `--stdin --raw-pcm` to read raw float32 little-endian PCM data from stdin. This is useful when integrating with pipelines that produce headerless PCM data.

**This mode does not require ffmpeg** - audio is read directly from stdin and resampled using scipy if needed.

```shell
# Raw PCM at 8kHz - no resampling needed (source matches default target)
ffmpeg -i input.mp3 -f f32le -ac 1 -ar 8000 pipe: | \
  audio-pattern-detector match --stdin --raw-pcm --source-sample-rate 8000 --pattern-file pattern.wav

# Raw PCM at 16kHz input, target 16kHz - no resampling needed (source matches target)
ffmpeg -i input.mp3 -f f32le -ac 1 -ar 16000 pipe: | \
  audio-pattern-detector match --stdin --raw-pcm --source-sample-rate 16000 --target-sample-rate 16000 --pattern-file pattern.wav

# Raw PCM at 16kHz input, target 8kHz (default) - resampled from 16kHz to 8kHz
some-16khz-source | \
  audio-pattern-detector match --stdin --raw-pcm --source-sample-rate 16000 --pattern-file pattern.wav
```

**Note**: When using `--stdin --raw-pcm`:
- Input must be raw float32 little-endian PCM (f32le)
- `--source-sample-rate` is required to specify the input sample rate
- Output is always JSONL format for real-time streaming
- **Resampling**: If `--source-sample-rate` differs from `--target-sample-rate` (default: 8000), audio is resampled using scipy
- Pattern files are loaded at the target sample rate (default: 8000)
- WAV header detection: If WAV data is accidentally sent to raw PCM mode, an error is raised with a helpful message

### Show-config - Show computed configuration for patterns

Use `show-config` to see computed configuration for pattern files without processing audio:

```shell
audio-pattern-detector show-config --pattern-folder ./clips
```

Output:
```json
{
  "default_seconds_per_chunk": 60,
  "min_chunk_size_seconds": 2,
  "sample_rate": 8000,
  "clips": {
    "pattern1": {
      "duration_seconds": 0.5,
      "sliding_window_seconds": 1,
      "is_pure_tone": false
    }
  }
}
```

- `default_seconds_per_chunk`: The default chunk size (60 seconds)
- `min_chunk_size_seconds`: Minimum required chunk size for these patterns (use this or higher with `--chunk-seconds`)

#### JSONL Streaming Output (match)

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

**Requires ffmpeg** to convert audio files.

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
