//! Numerical routines used by the detector: BS.1770 loudness, peak finding,
//! resampling, Pearson correlation and real spectra. FFT cross-correlation
//! comes from the `fft-correlation` crate.

pub mod loudness;
pub mod pearson;
pub mod peaks;
pub mod resample;
pub mod spectrum;

pub use loudness::{integrated_loudness, loudness_normalize};
pub use pearson::pearson_correlation_1d;
pub use peaks::{find_peaks_1d, FindPeaksOptions};
pub use resample::{resample_1d, resample_preserve_maxima_1d};
pub use spectrum::{hanning, rfft_magnitude, rfft_frequencies};
