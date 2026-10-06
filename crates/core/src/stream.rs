//! Audio sources that feed the detector with mono float32 samples.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::dsp::resample_1d;
use crate::error::{Error, Result};
use crate::wav::{mix_to_mono, read_full, SampleFormat, WavReader};

/// A pull-based source of mono float32 samples at a fixed sample rate.
pub trait SampleSource {
    /// Read up to `max_samples` samples. Returns fewer only at the end of the
    /// audio; an empty vector means EOF.
    fn read_samples(&mut self, max_samples: usize) -> Result<Vec<f32>>;
}

impl<S: SampleSource + ?Sized> SampleSource for &mut S {
    fn read_samples(&mut self, max_samples: usize) -> Result<Vec<f32>> {
        (**self).read_samples(max_samples)
    }
}

/// A named audio source to search for patterns in.
pub struct AudioStream<'a> {
    pub name: String,
    pub source: Box<dyn SampleSource + 'a>,
    pub sample_rate: u32,
}

impl<'a> AudioStream<'a> {
    pub fn new(name: impl Into<String>, source: impl SampleSource + 'a, sample_rate: u32) -> Self {
        Self { name: name.into(), source: Box::new(source), sample_rate }
    }

    /// Stream over audio that is already in memory.
    pub fn from_samples(name: impl Into<String>, samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new(name, MemorySource::new(samples), sample_rate)
    }
}

/// Resample audio with FFT-based resampling; a no-op when the rates match.
pub fn resample_audio(audio: Vec<f32>, orig_sr: u32, target_sr: u32) -> Vec<f32> {
    if orig_sr == target_sr {
        return audio;
    }
    let num_samples = (audio.len() as f64 * target_sr as f64 / orig_sr as f64) as usize;
    resample_1d(&audio, num_samples)
}

/// In-memory samples.
pub struct MemorySource {
    samples: Vec<f32>,
    position: usize,
}

impl MemorySource {
    pub fn new(samples: Vec<f32>) -> Self {
        Self { samples, position: 0 }
    }
}

impl SampleSource for MemorySource {
    fn read_samples(&mut self, max_samples: usize) -> Result<Vec<f32>> {
        let end = self.position.saturating_add(max_samples).min(self.samples.len());
        let out = self.samples[self.position..end].to_vec();
        self.position = end;
        Ok(out)
    }
}

/// Raw headerless float32 little-endian PCM (e.g. ffmpeg `-f f32le` output).
pub struct F32leSource<R: Read> {
    reader: R,
}

impl<R: Read> F32leSource<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<R: Read> SampleSource for F32leSource<R> {
    fn read_samples(&mut self, max_samples: usize) -> Result<Vec<f32>> {
        let mut buf = vec![0u8; max_samples * 4];
        let got = read_full(&mut self.reader, &mut buf)?;
        // A trailing partial sample is dropped.
        Ok(buf[..got].as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect())
    }
}

/// WAV file source. Mixes to mono and resamples to the target rate if needed,
/// so no ffmpeg is required.
pub struct WavFileSource {
    wav: WavReader<BufReader<File>>,
    target_sample_rate: u32,
    validated: bool,
}

impl WavFileSource {
    pub fn open(path: impl AsRef<Path>, target_sample_rate: u32) -> Result<Self> {
        let path = path.as_ref();
        let fail = |e: Error| Error::invalid(format!("Failed to read WAV file {}: {e}", path.display()));
        let file = File::open(path).map_err(|e| fail(e.into()))?;
        let wav = WavReader::new(BufReader::new(file)).map_err(fail)?;

        let channels = wav.spec().channels;
        if channels != 1 {
            eprintln!("Warning: WAV has {channels} channels, will be mixed to mono");
        }
        Ok(Self { wav, target_sample_rate, validated: false })
    }

    pub fn input_sample_rate(&self) -> u32 {
        self.wav.spec().sample_rate
    }

    pub fn needs_resample(&self) -> bool {
        self.input_sample_rate() != self.target_sample_rate
    }

    /// Check the first chunk for signs of corrupt audio.
    fn validate_first_chunk(&mut self, audio: &[f32]) {
        if self.validated || audio.is_empty() {
            return;
        }
        self.validated = true;

        if audio.iter().any(|v| v.is_nan()) {
            eprintln!("Warning: Audio contains NaN values - data may be corrupt");
        }
        if audio.iter().any(|v| v.is_infinite()) {
            eprintln!("Warning: Audio contains Inf values - data may be corrupt");
        }
        let max_abs = audio.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        if max_abs > 1.5 {
            eprintln!("Warning: Audio values exceed expected range (max: {max_abs:.2})");
        }
        if audio.iter().all(|&v| v == 0.0) {
            eprintln!("Warning: First chunk is all zeros - verify input is correct");
        }
    }
}

impl SampleSource for WavFileSource {
    fn read_samples(&mut self, max_samples: usize) -> Result<Vec<f32>> {
        let input_sample_rate = self.input_sample_rate();
        let input_frames = if self.needs_resample() {
            (max_samples as f64 * input_sample_rate as f64 / self.target_sample_rate as f64) as usize
        } else {
            max_samples
        };

        let frames = self.wav.read_frames(input_frames)?;
        if frames.is_empty() {
            return Ok(frames);
        }
        let audio = mix_to_mono(frames, self.wav.spec().channels);
        self.validate_first_chunk(&audio);
        Ok(resample_audio(audio, input_sample_rate, self.target_sample_rate))
    }
}

/// WAV read from a pipe (e.g. stdin). Requires mono audio already at the
/// target sample rate in 16-bit PCM, 32-bit PCM or 32-bit float; the declared
/// data length is ignored and audio is read until EOF.
pub struct WavStreamSource<R: Read> {
    wav: WavReader<R>,
}

impl<R: Read> WavStreamSource<R> {
    pub fn new(reader: R, target_sample_rate: u32) -> Result<Self> {
        let wav = WavReader::new(reader)?.read_until_eof();
        let spec = wav.spec();
        match spec.format {
            SampleFormat::I16 | SampleFormat::I32 | SampleFormat::F32 => {}
            SampleFormat::F64 => return Err(Error::invalid("Expected 32-bit float, got 64")),
            other => {
                return Err(Error::invalid(format!(
                    "Expected 16-bit or 32-bit PCM, got {}",
                    other.bytes_per_sample() * 8
                )))
            }
        }
        if spec.channels != 1 {
            return Err(Error::invalid(format!("Expected mono (1 channel), got {}", spec.channels)));
        }
        if spec.sample_rate != target_sample_rate {
            return Err(Error::invalid(format!(
                "Expected {target_sample_rate} Hz, got {}",
                spec.sample_rate
            )));
        }
        Ok(Self { wav })
    }

    pub fn format(&self) -> SampleFormat {
        self.wav.spec().format
    }
}

impl<R: Read> SampleSource for WavStreamSource<R> {
    fn read_samples(&mut self, max_samples: usize) -> Result<Vec<f32>> {
        self.wav.read_frames(max_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::encode_wav_i16;

    #[test]
    fn test_memory_source_reads_in_chunks() {
        let mut source = MemorySource::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(source.read_samples(2).unwrap(), vec![1.0, 2.0]);
        assert_eq!(source.read_samples(2).unwrap(), vec![3.0, 4.0]);
        assert_eq!(source.read_samples(2).unwrap(), vec![5.0]);
        assert_eq!(source.read_samples(2).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn test_f32le_source_decodes_and_drops_partial_sample() {
        let mut bytes: Vec<u8> = [0.5f32, -0.25, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        bytes.extend_from_slice(&[1, 2]);
        let mut source = F32leSource::new(bytes.as_slice());
        assert_eq!(source.read_samples(2).unwrap(), vec![0.5, -0.25]);
        assert_eq!(source.read_samples(2).unwrap(), vec![1.0]);
        assert_eq!(source.read_samples(2).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn test_wav_stream_source_validates_header() {
        let wav = encode_wav_i16(&[0.5, -0.5, 0.25], 8000);
        let mut source = WavStreamSource::new(wav.as_slice(), 8000).unwrap();
        assert_eq!(source.format(), SampleFormat::I16);
        assert_eq!(source.read_samples(10).unwrap(), vec![0.5, -0.5, 0.25]);

        let err = WavStreamSource::new(wav.as_slice(), 16000).err().unwrap().to_string();
        assert_eq!(err, "Expected 16000 Hz, got 8000");
    }

    #[test]
    fn test_resample_audio_same_rate_is_identity() {
        assert_eq!(resample_audio(vec![0.1, 0.2, 0.3], 8000, 8000), vec![0.1, 0.2, 0.3]);
        assert_eq!(resample_audio(vec![0.0; 16000], 16000, 8000).len(), 8000);
    }
}
