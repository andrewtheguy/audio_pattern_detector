use std::fmt;

/// Error type for all detector operations.
#[derive(Debug)]
pub enum Error {
    /// Invalid input, configuration or audio data.
    Invalid(String),
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// FFT cross-correlation failure.
    Correlation(fft_correlation::FftCorrelationError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Error::Invalid(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(message) => f.write_str(message),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Correlation(e) => write!(f, "correlation error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Invalid(_) => None,
            Error::Io(e) => Some(e),
            Error::Correlation(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<fft_correlation::FftCorrelationError> for Error {
    fn from(e: fft_correlation::FftCorrelationError) -> Self {
        Error::Correlation(e)
    }
}
