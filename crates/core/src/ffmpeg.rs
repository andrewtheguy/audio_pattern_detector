//! ffmpeg subprocess helpers for decoding non-WAV audio.

use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::OnceLock;

use crate::error::{Error, Result};
use crate::stream::{F32leSource, SampleSource};

/// Check whether ffmpeg is available on the system (cached after the first call).
pub fn is_ffmpeg_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("ffmpeg")
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    })
}

/// ffmpeg process decoding a file to mono float32 PCM at the target sample rate.
pub struct FfmpegSource {
    child: Child,
    source: F32leSource<ChildStdout>,
}

impl FfmpegSource {
    pub fn open(path: impl AsRef<Path>, target_sample_rate: u32) -> Result<Self> {
        let path = path.as_ref();
        if !is_ffmpeg_available() {
            return Err(Error::invalid(format!(
                "ffmpeg not available and file {} is not a WAV file. \
                 Install ffmpeg or use WAV files.",
                path.display()
            )));
        }
        let mut child = Command::new("ffmpeg")
            .arg("-i")
            .arg(path)
            .args(["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar"])
            .arg(target_sample_rate.to_string())
            .args(["-loglevel", "error", "pipe:"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().expect("stdout is piped");
        Ok(Self { child, source: F32leSource::new(stdout) })
    }

    /// Wait for ffmpeg to exit and fail if it reported an error.
    pub fn finish(mut self) -> Result<()> {
        let status = self.child.wait()?;
        if !status.success() {
            return Err(Error::invalid(format!(
                "ffmpeg command failed with return code {}",
                status.code().map_or_else(|| "unknown".to_string(), |c| c.to_string())
            )));
        }
        Ok(())
    }
}

impl SampleSource for FfmpegSource {
    fn read_samples(&mut self, max_samples: usize) -> Result<Vec<f32>> {
        self.source.read_samples(max_samples)
    }
}

impl Drop for FfmpegSource {
    fn drop(&mut self) {
        // Reap the process if the caller bailed out before `finish`.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Decode a whole audio file with ffmpeg to mono float32 at the target sample rate.
pub fn load_audio_ffmpeg(path: impl AsRef<Path>, target_sample_rate: u32) -> Result<Vec<f32>> {
    const READ_SAMPLES: usize = 1 << 16;
    let mut source = FfmpegSource::open(path, target_sample_rate)?;
    let mut samples = Vec::new();
    loop {
        let chunk = source.read_samples(READ_SAMPLES)?;
        if chunk.is_empty() {
            break;
        }
        samples.extend_from_slice(&chunk);
    }
    source.finish()?;
    Ok(samples)
}
