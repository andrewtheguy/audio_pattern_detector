//! Helpers shared by the integration tests.
#![allow(dead_code)]

use audio_pattern_detector_core::{AudioClip, AudioStream, DEFAULT_TARGET_SAMPLE_RATE};

pub const SR: u32 = DEFAULT_TARGET_SAMPLE_RATE;

/// Sine wave with unit amplitude, `floor(sample_rate * duration)` samples long.
pub fn sine_tone(frequency: f64, duration: f64, sample_rate: u32) -> Vec<f32> {
    let n = (sample_rate as f64 * duration) as usize;
    let step = duration / n as f64;
    (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * frequency * (i as f64 * step)).sin() as f32)
        .collect()
}

/// Silence, `floor(sample_rate * duration)` samples long.
pub fn silence(duration: f64, sample_rate: u32) -> Vec<f32> {
    vec![0.0; (sample_rate as f64 * duration) as usize]
}

/// Concatenate audio segments.
pub fn concat(parts: &[&[f32]]) -> Vec<f32> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

/// Copy `clip` over `audio` starting at sample `start`.
pub fn insert_at(audio: &mut [f32], start: usize, clip: &[f32]) {
    audio[start..start + clip.len()].copy_from_slice(clip);
}

/// In-memory audio stream at the default sample rate.
pub fn stream_from_samples(name: &str, audio: &[f32]) -> AudioStream<'static> {
    AudioStream::from_samples(name, audio.to_vec(), SR)
}

/// Pattern clip at the default sample rate.
pub fn clip_from_samples(name: &str, audio: &[f32]) -> AudioClip {
    AudioClip::new(name, audio.to_vec(), SR)
}

/// Small deterministic PRNG (xorshift64*) so tests need no extra dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Gaussian noise with the given standard deviation.
    pub fn noise(&mut self, samples: usize, std_dev: f64) -> Vec<f32> {
        (0..samples).map(|_| (self.normal() * std_dev) as f32).collect()
    }
}
