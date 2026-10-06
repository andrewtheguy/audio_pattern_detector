//! Minimal WAV reading and writing (no external decoder needed).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::{Error, Result};

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
/// Size of a `WAVEFORMATEXTENSIBLE` fmt chunk, the largest one parsed.
const MAX_FMT_BYTES: usize = 40;

/// Sample encoding of a WAV data chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    U8,
    I16,
    I24,
    I32,
    F32,
    F64,
}

impl SampleFormat {
    pub fn bytes_per_sample(self) -> usize {
        match self {
            SampleFormat::U8 => 1,
            SampleFormat::I16 => 2,
            SampleFormat::I24 => 3,
            SampleFormat::I32 | SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
        }
    }

    /// Short name used in diagnostics, e.g. `int16` or `float32`.
    pub fn name(self) -> &'static str {
        match self {
            SampleFormat::U8 => "uint8",
            SampleFormat::I16 => "int16",
            SampleFormat::I24 => "int24",
            SampleFormat::I32 => "int32",
            SampleFormat::F32 => "float32",
            SampleFormat::F64 => "float64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavSpec {
    pub format: SampleFormat,
    pub channels: u16,
    pub sample_rate: u32,
}

/// Read until `buf` is full or EOF; returns the number of bytes read.
pub(crate) fn read_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

fn skip_bytes<R: Read>(reader: &mut R, count: u64) -> Result<()> {
    let skipped = std::io::copy(&mut reader.by_ref().take(count), &mut std::io::sink())?;
    if skipped != count {
        return Err(Error::invalid("WAV file truncated while skipping chunk"));
    }
    Ok(())
}

/// Streaming WAV reader: parses the header up front, then yields frames
/// converted to float32 in [-1, 1].
pub struct WavReader<R: Read> {
    reader: R,
    spec: WavSpec,
    /// Bytes left in the data chunk, or `None` to read until EOF.
    remaining: Option<u64>,
}

impl<R: Read> WavReader<R> {
    /// Parse the RIFF/WAVE header and position the reader at the first sample.
    pub fn new(mut reader: R) -> Result<Self> {
        let mut tag = [0u8; 4];
        let n = read_full(&mut reader, &mut tag)?;
        if n < 4 || &tag != b"RIFF" {
            return Err(Error::invalid(format!(
                "Not a WAV file: expected RIFF, got {:?}",
                String::from_utf8_lossy(&tag[..n])
            )));
        }
        let mut size = [0u8; 4];
        read_full(&mut reader, &mut size)?; // file size (ignored)
        let n = read_full(&mut reader, &mut tag)?;
        if n < 4 || &tag != b"WAVE" {
            return Err(Error::invalid(format!(
                "Not a WAV file: expected WAVE, got {:?}",
                String::from_utf8_lossy(&tag[..n])
            )));
        }

        let mut spec: Option<WavSpec> = None;
        loop {
            if read_full(&mut reader, &mut tag)? < 4 {
                return Err(Error::invalid(if spec.is_some() {
                    "WAV file missing data chunk"
                } else {
                    "WAV file missing fmt chunk"
                }));
            }
            if read_full(&mut reader, &mut size)? < 4 {
                return Err(Error::invalid("WAV file truncated"));
            }
            let chunk_size = u32::from_le_bytes(size);

            match &tag {
                b"fmt " => {
                    // Only the start of the chunk is parsed; never allocate
                    // from the declared size.
                    let mut fmt = vec![0u8; (chunk_size as usize).min(MAX_FMT_BYTES)];
                    if read_full(&mut reader, &mut fmt)? < fmt.len() || fmt.len() < 16 {
                        return Err(Error::invalid("WAV fmt chunk too short"));
                    }
                    spec = Some(parse_fmt(&fmt)?);
                    let unread = chunk_size as u64 - fmt.len() as u64;
                    skip_bytes(&mut reader, unread + (chunk_size % 2) as u64)?;
                }
                b"data" => {
                    let spec = spec.ok_or_else(|| Error::invalid("WAV file missing fmt chunk"))?;
                    // Streaming encoders write 0 or 0xFFFFFFFF when the length is unknown.
                    let remaining = match chunk_size {
                        0 | u32::MAX => None,
                        n => Some(n as u64),
                    };
                    return Ok(Self { reader, spec, remaining });
                }
                _ => skip_bytes(&mut reader, chunk_size as u64 + (chunk_size % 2) as u64)?,
            }
        }
    }

    pub fn spec(&self) -> WavSpec {
        self.spec
    }

    /// Ignore the declared data chunk length and read until EOF (for pipes).
    pub fn read_until_eof(mut self) -> Self {
        self.remaining = None;
        self
    }

    /// Read up to `max_frames` frames as interleaved float32 samples.
    ///
    /// Returns fewer frames only at the end of the data; an empty vector means EOF.
    pub fn read_frames(&mut self, max_frames: usize) -> Result<Vec<f32>> {
        let frame_bytes = self.spec.format.bytes_per_sample() * self.spec.channels as usize;
        let mut want = max_frames.saturating_mul(frame_bytes);
        if let Some(remaining) = self.remaining {
            want = want.min(usize::try_from(remaining).unwrap_or(usize::MAX));
        }
        let mut buf = vec![0u8; want];
        let got = read_full(&mut self.reader, &mut buf)?;
        if let Some(remaining) = self.remaining.as_mut() {
            *remaining -= got as u64;
        }
        // Drop a trailing partial frame.
        buf.truncate(got - got % frame_bytes);
        Ok(decode_samples(&buf, self.spec.format))
    }

    /// Read all remaining frames as interleaved float32 samples.
    pub fn read_all_frames(&mut self) -> Result<Vec<f32>> {
        let frame_bytes = self.spec.format.bytes_per_sample() * self.spec.channels as usize;
        // `read_to_end` grows the buffer as data arrives, so a bogus declared
        // length cannot trigger a huge allocation.
        let mut buf = Vec::new();
        match self.remaining.as_mut() {
            Some(remaining) => {
                let got = self.reader.by_ref().take(*remaining).read_to_end(&mut buf)?;
                *remaining -= got as u64;
            }
            None => {
                self.reader.read_to_end(&mut buf)?;
            }
        }
        // Drop a trailing partial frame.
        buf.truncate(buf.len() - buf.len() % frame_bytes);
        Ok(decode_samples(&buf, self.spec.format))
    }
}

fn parse_fmt(fmt: &[u8]) -> Result<WavSpec> {
    let u16_at = |i: usize| u16::from_le_bytes([fmt[i], fmt[i + 1]]);
    let mut audio_format = u16_at(0);
    let channels = u16_at(2);
    let sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
    let bits_per_sample = u16_at(14);

    if audio_format == WAVE_FORMAT_EXTENSIBLE {
        if fmt.len() < 26 {
            return Err(Error::invalid("WAV extensible fmt chunk too short"));
        }
        audio_format = u16_at(24);
    }

    let format = match (audio_format, bits_per_sample) {
        (WAVE_FORMAT_PCM, 8) => SampleFormat::U8,
        (WAVE_FORMAT_PCM, 16) => SampleFormat::I16,
        (WAVE_FORMAT_PCM, 24) => SampleFormat::I24,
        (WAVE_FORMAT_PCM, 32) => SampleFormat::I32,
        (WAVE_FORMAT_IEEE_FLOAT, 32) => SampleFormat::F32,
        (WAVE_FORMAT_IEEE_FLOAT, 64) => SampleFormat::F64,
        (WAVE_FORMAT_PCM, bits) => {
            return Err(Error::invalid(format!("Unsupported PCM sample width: {bits} bits")))
        }
        (WAVE_FORMAT_IEEE_FLOAT, bits) => {
            return Err(Error::invalid(format!("Unsupported float sample width: {bits} bits")))
        }
        (other, _) => {
            return Err(Error::invalid(format!(
                "Expected PCM (1) or IEEE float (3) format, got {other}"
            )))
        }
    };
    if channels == 0 {
        return Err(Error::invalid("WAV file declares 0 channels"));
    }
    if sample_rate == 0 {
        return Err(Error::invalid("WAV file declares a sample rate of 0"));
    }

    Ok(WavSpec { format, channels, sample_rate })
}

/// Convert raw little-endian sample bytes to float32 in [-1, 1].
fn decode_samples(bytes: &[u8], format: SampleFormat) -> Vec<f32> {
    match format {
        SampleFormat::U8 => bytes.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        SampleFormat::I16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| i16::from_le_bytes(b) as f32 / 32768.0)
            .collect(),
        SampleFormat::I24 => bytes
            .as_chunks::<3>()
            .0
            .iter()
            // Place the 24 bits in the top of an i32 so /2^31 normalizes correctly.
            .map(|b| i32::from_le_bytes([0, b[0], b[1], b[2]]) as f32 / 2147483648.0)
            .collect(),
        SampleFormat::I32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| i32::from_le_bytes(b) as f32 / 2147483648.0)
            .collect(),
        SampleFormat::F32 => bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
        SampleFormat::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| f64::from_le_bytes(b) as f32)
            .collect(),
    }
}

/// Average interleaved channels down to mono.
pub fn mix_to_mono(interleaved: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return interleaved;
    }
    let channels = channels as usize;
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn load_wav<R: Read>(reader: R, source_name: &str) -> Result<(Vec<f32>, u32)> {
    let read = || -> Result<(Vec<f32>, u32)> {
        let mut wav = WavReader::new(reader)?;
        let spec = wav.spec();
        let samples = wav.read_all_frames()?;
        Ok((mix_to_mono(samples, spec.channels), spec.sample_rate))
    };
    read().map_err(|e| Error::invalid(format!("Failed to read WAV data from {source_name}: {e}")))
}

/// Load a WAV file as mono float32 in [-1, 1]. Returns `(samples, sample_rate)`.
pub fn load_wav_file(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let source_name = format!("file {}", path.display());
    let file = File::open(path)
        .map_err(|e| Error::invalid(format!("Failed to read WAV data from {source_name}: {e}")))?;
    load_wav(BufReader::new(file), &source_name)
}

/// Load WAV data from memory as mono float32 in [-1, 1]. Returns `(samples, sample_rate)`.
pub fn load_wav_from_bytes(wav_bytes: &[u8], name: &str) -> Result<(Vec<f32>, u32)> {
    load_wav(wav_bytes, name)
}

/// Encode mono float32 audio in [-1, 1] as a 16-bit PCM WAV file in memory.
pub fn encode_wav_i16(audio: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (audio.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + audio.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &sample in audio {
        let quantized = (sample * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&quantized.to_le_bytes());
    }
    out
}

/// Write mono float32 audio in [-1, 1] to a 16-bit PCM WAV file.
pub fn write_wav_file(path: impl AsRef<Path>, audio: &[f32], sample_rate: u32) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&encode_wav_i16(audio, sample_rate))?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(format_tag: u16, channels: u16, sample_rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&format_tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        let block_align = channels * bits / 8;
        out.extend_from_slice(&(sample_rate * block_align as u32).to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn test_decode_int16() {
        let data: Vec<u8> = [0i16, 16384, -32768, 32767].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (samples, sr) = load_wav_from_bytes(&wav_bytes(1, 1, 8000, 16, &data), "t").unwrap();
        assert_eq!(sr, 8000);
        assert_eq!(samples, vec![0.0, 0.5, -1.0, 32767.0 / 32768.0]);
    }

    #[test]
    fn test_decode_uint8_int24_int32_float() {
        let (samples, _) = load_wav_from_bytes(&wav_bytes(1, 1, 8000, 8, &[128, 255, 0, 192]), "t").unwrap();
        assert_eq!(samples, vec![0.0, 127.0 / 128.0, -1.0, 0.5]);

        let data = [0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0xFF, 0xFF, 0x7F];
        let (samples, _) = load_wav_from_bytes(&wav_bytes(1, 1, 8000, 24, &data), "t").unwrap();
        assert_eq!(samples, vec![0.5, -1.0, 8388607.0 / 8388608.0]);

        let data: Vec<u8> = [1073741824i32, i32::MIN].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (samples, _) = load_wav_from_bytes(&wav_bytes(1, 1, 8000, 32, &data), "t").unwrap();
        assert_eq!(samples, vec![0.5, -1.0]);

        let data: Vec<u8> = [0.25f32, -0.75].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (samples, _) = load_wav_from_bytes(&wav_bytes(3, 1, 16000, 32, &data), "t").unwrap();
        assert_eq!(samples, vec![0.25, -0.75]);

        let data: Vec<u8> = [0.25f64, -0.75].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (samples, sr) = load_wav_from_bytes(&wav_bytes(3, 1, 16000, 64, &data), "t").unwrap();
        assert_eq!((samples, sr), (vec![0.25, -0.75], 16000));
    }

    #[test]
    fn test_stereo_is_mixed_to_mono() {
        let data: Vec<u8> = [16384i16, 0, -16384, -16384].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (samples, _) = load_wav_from_bytes(&wav_bytes(1, 2, 8000, 16, &data), "t").unwrap();
        assert_eq!(samples, vec![0.25, -0.5]);
    }

    #[test]
    fn test_oversized_fmt_chunk_does_not_allocate_declared_size() {
        let data: Vec<u8> = [16384i16, -16384].iter().flat_map(|v| v.to_le_bytes()).collect();
        let plain = wav_bytes(1, 1, 8000, 16, &data);

        // A fmt chunk declaring 4 GiB with nothing behind it is a truncated file.
        let mut bytes = plain[..36].to_vec();
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = load_wav_from_bytes(&bytes, "huge").unwrap_err().to_string();
        assert_eq!(err, "Failed to read WAV data from huge: WAV fmt chunk too short");

        // Extension bytes beyond the parsed part (and the pad byte) are skipped.
        let mut bytes = plain[..16].to_vec();
        bytes.extend_from_slice(&45u32.to_le_bytes());
        bytes.extend_from_slice(&plain[20..36]);
        bytes.extend_from_slice(&[0u8; 29 + 1]);
        bytes.extend_from_slice(&plain[36..]);
        let (samples, sr) = load_wav_from_bytes(&bytes, "extended").unwrap();
        assert_eq!((samples, sr), (vec![0.5, -0.5], 8000));
    }

    #[test]
    fn test_skips_unknown_chunks_and_honours_data_length() {
        let data: Vec<u8> = [16384i16, -16384].iter().flat_map(|v| v.to_le_bytes()).collect();
        let plain = wav_bytes(1, 1, 8000, 16, &data);
        // Insert an odd-sized LIST chunk (plus pad byte) before data, and trailing junk after it.
        let mut bytes = plain[..36].to_vec();
        bytes.extend_from_slice(b"LIST");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 0]);
        bytes.extend_from_slice(&plain[36..]);
        bytes.extend_from_slice(&[9, 9, 9, 9]);
        let (samples, _) = load_wav_from_bytes(&bytes, "t").unwrap();
        assert_eq!(samples, vec![0.5, -0.5]);
    }

    #[test]
    fn test_read_until_eof_ignores_data_length() {
        let data: Vec<u8> = [16384i16, -16384, 8192].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut bytes = wav_bytes(1, 1, 8000, 16, &data);
        bytes[40..44].copy_from_slice(&2u32.to_le_bytes());
        let mut reader = WavReader::new(bytes.as_slice()).unwrap().read_until_eof();
        assert_eq!(reader.read_frames(2).unwrap(), vec![0.5, -0.5]);
        assert_eq!(reader.read_frames(2).unwrap(), vec![0.25]);
        assert_eq!(reader.read_frames(2).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn test_unknown_data_length_reads_to_eof() {
        // Three samples plus a trailing partial frame, which is dropped.
        let mut data: Vec<u8> = [16384i16, -16384, 8192].iter().flat_map(|v| v.to_le_bytes()).collect();
        data.push(7);
        for unknown_size in [u32::MAX, 0] {
            let mut bytes = wav_bytes(1, 1, 8000, 16, &data);
            bytes[40..44].copy_from_slice(&unknown_size.to_le_bytes());
            let (samples, sr) = load_wav_from_bytes(&bytes, "t").unwrap();
            assert_eq!((samples, sr), (vec![0.5, -0.5, 0.25], 8000));
        }
    }

    #[test]
    fn test_declared_data_length_larger_than_file() {
        let data: Vec<u8> = [16384i16, -16384].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut bytes = wav_bytes(1, 1, 8000, 16, &data);
        bytes[40..44].copy_from_slice(&(u32::MAX - 1).to_le_bytes());
        let mut reader = WavReader::new(bytes.as_slice()).unwrap();
        assert_eq!(reader.read_all_frames().unwrap(), vec![0.5, -0.5]);
        assert_eq!(reader.read_all_frames().unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn test_invalid_headers_are_rejected() {
        let err = load_wav_from_bytes(b"not a wav file at all", "junk").unwrap_err().to_string();
        assert_eq!(err, "Failed to read WAV data from junk: Not a WAV file: expected RIFF, got \"not \"");

        let err = load_wav_from_bytes(&wav_bytes(7, 1, 8000, 8, &[]), "mulaw").unwrap_err().to_string();
        assert_eq!(
            err,
            "Failed to read WAV data from mulaw: Expected PCM (1) or IEEE float (3) format, got 7"
        );

        let err = load_wav_from_bytes(&wav_bytes(1, 1, 8000, 16, &[])[..36], "nodata").unwrap_err().to_string();
        assert_eq!(err, "Failed to read WAV data from nodata: WAV file missing data chunk");
    }

    #[test]
    fn test_encode_roundtrip() {
        let audio = [0.0_f32, 0.5, -0.5, 1.0, -1.0, 0.25];
        let (samples, sr) = load_wav_from_bytes(&encode_wav_i16(&audio, 8000), "t").unwrap();
        assert_eq!(sr, 8000);
        assert_eq!(samples, vec![0.0, 0.5, -0.5, 32767.0 / 32768.0, -1.0, 0.25]);
    }
}
