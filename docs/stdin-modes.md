# Stdin Modes

This document covers all stdin-based input modes for streaming audio processing.

**None of these modes require ffmpeg** - audio is processed using Python's `wave` module and scipy.

## Stdin Mode (WAV)

Use `--stdin` to read WAV format audio from stdin. This mode always outputs JSONL for real-time streaming detection. The sample rate is automatically read from the WAV header.

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

## Stdin Mode (Raw PCM)

Use `--stdin --raw-pcm` to read raw float32 little-endian PCM data from stdin. This is useful when integrating with pipelines that produce headerless PCM data.

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
  [remaining bytes until EOF] audio data (WAV format, or raw PCM with --raw-pcm)
```

### Usage

```shell
# WAV audio stream (default)
audio-pattern-detector match --multiplexed-stdin < payload.bin

# Raw PCM audio stream
audio-pattern-detector match --multiplexed-stdin --raw-pcm --source-sample-rate 16000 < payload.bin
```

### Node.js Example

```javascript
const { spawn } = require('child_process');
const fs = require('fs');

function buildMultiplexedPayload(patterns, audioData) {
  const chunks = [];

  // Number of patterns (uint32 LE)
  const countBuf = Buffer.alloc(4);
  countBuf.writeUInt32LE(patterns.length);
  chunks.push(countBuf);

  // Each pattern: name_len + name + data_len + wav_data
  for (const { name, wavData } of patterns) {
    const nameBuf = Buffer.from(name, 'utf8');
    const nameLenBuf = Buffer.alloc(4);
    nameLenBuf.writeUInt32LE(nameBuf.length);
    const dataLenBuf = Buffer.alloc(4);
    dataLenBuf.writeUInt32LE(wavData.length);

    chunks.push(nameLenBuf, nameBuf, dataLenBuf, wavData);
  }

  // Audio stream
  chunks.push(audioData);
  return Buffer.concat(chunks);
}

// Build payload with multiple patterns
const patterns = [
  { name: 'beep', wavData: fs.readFileSync('patterns/beep.wav') },
  { name: 'tone', wavData: fs.readFileSync('patterns/tone.wav') },
];
const audioData = fs.readFileSync('input.wav');
const payload = buildMultiplexedPayload(patterns, audioData);

// Spawn detector
const detector = spawn('audio-pattern-detector', ['match', '--multiplexed-stdin']);
detector.stdin.end(payload);

// Parse JSONL output
detector.stdout.on('data', (data) => {
  data.toString().trim().split('\n').forEach(line => {
    const event = JSON.parse(line);
    if (event.type === 'pattern_detected') {
      console.log(`Detected ${event.clip_name} at ${event.timestamp_formatted}`);
    }
  });
});
```

### Python Example

```python
import subprocess
import struct
import json

def build_multiplexed_payload(patterns: list[tuple[str, bytes]], audio_data: bytes) -> bytes:
    payload = bytearray()

    # Number of patterns
    payload.extend(struct.pack('<I', len(patterns)))

    # Each pattern
    for name, wav_data in patterns:
        name_bytes = name.encode('utf-8')
        payload.extend(struct.pack('<I', len(name_bytes)))
        payload.extend(name_bytes)
        payload.extend(struct.pack('<I', len(wav_data)))
        payload.extend(wav_data)

    # Audio stream
    payload.extend(audio_data)
    return bytes(payload)

# Build payload
patterns = [
    ('beep', open('patterns/beep.wav', 'rb').read()),
    ('tone', open('patterns/tone.wav', 'rb').read()),
]
audio_data = open('input.wav', 'rb').read()
payload = build_multiplexed_payload(patterns, audio_data)

# Run detector
proc = subprocess.run(
    ['audio-pattern-detector', 'match', '--multiplexed-stdin'],
    input=payload,
    capture_output=True,
)

# Parse JSONL output
for line in proc.stdout.decode().strip().split('\n'):
    event = json.loads(line)
    print(event)
```

### Go Example

```go
package main

import (
    "bytes"
    "encoding/binary"
    "fmt"
    "os"
    "os/exec"
)

func buildPayload(patterns []struct{ Name string; Data []byte }, audioData []byte) []byte {
    var buf bytes.Buffer

    // Number of patterns
    binary.Write(&buf, binary.LittleEndian, uint32(len(patterns)))

    // Each pattern
    for _, p := range patterns {
        nameBytes := []byte(p.Name)
        binary.Write(&buf, binary.LittleEndian, uint32(len(nameBytes)))
        buf.Write(nameBytes)
        binary.Write(&buf, binary.LittleEndian, uint32(len(p.Data)))
        buf.Write(p.Data)
    }

    // Audio stream
    buf.Write(audioData)
    return buf.Bytes()
}

func main() {
    beepData, _ := os.ReadFile("patterns/beep.wav")
    audioData, _ := os.ReadFile("input.wav")

    patterns := []struct{ Name string; Data []byte }{
        {"beep", beepData},
    }

    payload := buildPayload(patterns, audioData)

    cmd := exec.Command("audio-pattern-detector", "match", "--multiplexed-stdin")
    cmd.Stdin = bytes.NewReader(payload)
    output, _ := cmd.Output()

    fmt.Println(string(output))
}
```

**Note**: When using `--multiplexed-stdin`:
- Patterns are sent as WAV data in the binary protocol (not file paths)
- Does not require `--pattern-file` or `--pattern-folder`
- Audio stream can be WAV (default) or raw PCM (with `--raw-pcm --source-sample-rate`)
- Output is always JSONL format with `{"type": "start", "source": "multiplexed-stdin"}` as the first event

## JSONL Output Format

All stdin modes output JSONL (one JSON object per line):

```jsonl
{"type": "start", "source": "stdin"}
{"type": "pattern_detected", "clip_name": "pattern", "timestamp": 5.5, "timestamp_formatted": "00:00:05.500"}
{"type": "end", "total_time": 60.0, "total_time_formatted": "00:01:00.000"}
```

Event types:
- `start` - Emitted when processing begins
- `pattern_detected` - Emitted each time a pattern is detected
- `end` - Emitted when processing completes
