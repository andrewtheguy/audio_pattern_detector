# Stdin Modes

This document covers all stdin-based input modes for streaming audio processing.

**None of these modes require ffmpeg** - audio is processed using manual WAV header parsing with proper format detection.

## Stdin Mode (WAV)

Use `--stdin` to read WAV format audio from stdin. This mode always outputs JSONL for real-time streaming detection. The WAV must be mono at the target sample rate (default: 8000 Hz).

Accepted WAV formats: 16-bit PCM, 32-bit PCM, or 32-bit IEEE float. When the input is already float32, no extra conversion is performed.

```shell
# WAV stdin at 8kHz mono (default target)
ffmpeg -i input.mp3 -f wav -acodec pcm_s16le -ac 1 -ar 8000 pipe: | \
  audio-pattern-detector match --stdin --pattern-file pattern.wav

# WAV stdin at 16kHz mono with custom target
ffmpeg -i input.mp3 -f wav -acodec pcm_s16le -ac 1 -ar 16000 pipe: | \
  audio-pattern-detector match --stdin --target-sample-rate 16000 --pattern-file pattern.wav

# WAV stdin with float32 encoding (passed through without conversion)
ffmpeg -i input.mp3 -f wav -acodec pcm_f32le -ac 1 -ar 8000 pipe: | \
  audio-pattern-detector match --stdin --pattern-file pattern.wav
```

**Note**: When using `--stdin`:
- Input must be WAV format (mono, at the target sample rate)
- Supported encodings: 16-bit PCM, 32-bit PCM, 32-bit IEEE float
- Float32 input is passed through without extra conversion
- Output is always JSONL format for real-time streaming

## Multiplexed Stdin Mode (for IPC)

Use `--multiplexed-stdin` to pipe both pattern files and audio through a single stdin stream. This is designed for interprocess communication from programs like Node.js, Go, Rust, etc.

This mode does not require `--pattern-file` or `--pattern-folder` (patterns are sent via stdin).

### Protocol Format

All integers are **uint32 little-endian**.

```
HEADER:
  [4 bytes] number_of_patterns

FOR EACH PATTERN:
  [4 bytes] name_length
  [name_length bytes] name (UTF-8 string)
  [4 bytes] data_length
  [data_length bytes] WAV file data

AUDIO STREAM:
  [remaining bytes until EOF] audio data (WAV format)
```

### Usage

```shell
audio-pattern-detector match --multiplexed-stdin < payload.bin
```

### Node.js Example

Patterns are written synchronously (they are small), then the audio file is
piped in from a read stream so the full payload is never held in memory.
Output is parsed line-by-line with `readline` so JSONL events are handled as
soon as they arrive.

```javascript
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const readline = require('node:readline');

function writeUInt32LE(stream, value) {
  const buf = Buffer.alloc(4);
  buf.writeUInt32LE(value);
  stream.write(buf);
}

function writePatternHeader(stream, patterns) {
  writeUInt32LE(stream, patterns.length);
  for (const { name, wavPath } of patterns) {
    const nameBuf = Buffer.from(name, 'utf8');
    const { size: wavSize } = fs.statSync(wavPath);
    writeUInt32LE(stream, nameBuf.length);
    stream.write(nameBuf);
    writeUInt32LE(stream, wavSize);
    // Patterns are small — read once and write in one go.
    stream.write(fs.readFileSync(wavPath));
  }
}

const patterns = [
  { name: 'beep', wavPath: 'patterns/beep.wav' },
  { name: 'tone', wavPath: 'patterns/tone.wav' },
];

const detector = spawn('audio-pattern-detector', ['match', '--multiplexed-stdin']);

writePatternHeader(detector.stdin, patterns);

// Stream the audio file so it never sits fully in memory. `end: true`
// closes stdin when the file ends, signalling EOF to the detector.
fs.createReadStream('input.wav').pipe(detector.stdin);

readline.createInterface({ input: detector.stdout }).on('line', (line) => {
  if (!line) return;
  const event = JSON.parse(line);
  if (event.type === 'pattern_detected') {
    console.log(`Detected ${event.clip_name} at ${event.timestamp_formatted}`);
  }
});

detector.on('exit', (code) => {
  if (code !== 0) process.exit(code ?? 1);
});
```

For a live capture (e.g. ffmpeg transcoding a radio stream to WAV), replace
`fs.createReadStream('input.wav')` with the ffmpeg child process's stdout:

```javascript
const ffmpeg = spawn('ffmpeg', [
  '-i', 'https://example.com/stream.m3u8',
  '-f', 'wav', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '8000',
  'pipe:',
]);
ffmpeg.stdout.pipe(detector.stdin);
```

**Note**: When using `--multiplexed-stdin`:
- Patterns are sent as WAV data in the binary protocol (not file paths)
- Does not require `--pattern-file` or `--pattern-folder`
- Audio stream must be WAV format (mono, at target sample rate)
- Output is always JSONL format with `{"type": "start", "source": "multiplexed-stdin"}` as the first event

## JSONL Output Format

All stdin modes output JSONL (one JSON object per line). By default, timestamp
events include both millisecond and formatted fields. Use `--timestamp-format
ms` or `--timestamp-format formatted` to limit the output to one form:

```jsonl
{"type": "start", "source": "stdin"}
{"type": "pattern_detected", "clip_name": "pattern", "timestamp_ms": 5500, "timestamp_formatted": "00:00:05.500"}
{"type": "end", "total_time_ms": 60000, "total_time_formatted": "00:01:00.000"}
```

Event types:
- `start` - Emitted when processing begins
- `pattern_detected` - Emitted each time a pattern is detected
- `end` - Emitted when processing completes
