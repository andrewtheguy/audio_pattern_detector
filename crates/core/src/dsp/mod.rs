//! Numerical routines used by the detector: FFT cross-correlation, BS.1770
//! loudness, peak finding, resampling, Pearson correlation and real spectra.

pub mod correlate;
pub mod loudness;
pub mod pearson;
pub mod peaks;
pub mod resample;
pub mod spectrum;

pub use correlate::{fft_correlate_full, CorrelationTemplate, CorrelationWorkspace};
pub use loudness::{integrated_loudness, loudness_normalize};
pub use pearson::pearson_correlation_1d;
pub use peaks::{find_peaks_1d, FindPeaksOptions};
pub use resample::{resample_1d, resample_preserve_maxima_1d};
pub use spectrum::{hanning, rfft_magnitude, rfft_frequencies};
