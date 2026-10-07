//! Python bindings (PyO3), enabled by the `python` feature. See `docs/python.md`.

use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::audio_clip::{AudioClip, DEFAULT_TARGET_SAMPLE_RATE};
use crate::detector::{
    AudioPatternDetector, DetectorOptions, PatternDetectedCallback, PeakTimes, DEFAULT_SECONDS_PER_CHUNK,
};
use crate::error::{Error, Result};
use crate::matching::{find_clips_in_file, find_clips_in_wav_stream, load_pattern_clips};
use crate::pattern_config::APD_EXTENSION;

/// First Python exception raised by a callback or stream during a match.
type PendingError = Mutex<Option<PyErr>>;

fn to_py_err(error: Error) -> PyErr {
    match error {
        Error::Invalid(message) => PyValueError::new_err(message),
        Error::Io(e) => e.into(),
        Error::Correlation(e) => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Keep the first Python exception; it is re-raised when the match returns.
fn store_error(pending: &PendingError, error: PyErr) {
    pending.lock().unwrap().get_or_insert(error);
}

/// Adapts a Python binary file-like object (anything with `read(n) -> bytes`).
struct PyReader<'a> {
    stream: &'a Py<PyAny>,
    pending: &'a PendingError,
}

impl Read for PyReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        // Stop pulling audio once a callback has failed.
        if self.pending.lock().unwrap().is_some() {
            return Err(io::Error::other("Python callback failed"));
        }
        let read = Python::attach(|py| -> PyResult<usize> {
            py.check_signals()?;
            let data = self.stream.bind(py).call_method1("read", (buf.len(),))?;
            let data = data
                .cast::<PyBytes>()
                .map_err(|_| PyTypeError::new_err("stream.read() must return bytes"))?
                .as_bytes();
            if data.len() > buf.len() {
                return Err(PyValueError::new_err("stream.read() returned more bytes than requested"));
            }
            buf[..data.len()].copy_from_slice(data);
            Ok(data.len())
        });
        read.map_err(|error| {
            store_error(self.pending, error);
            io::Error::other("Python stream read failed")
        })
    }
}

/// Result of matching one audio source.
#[pyclass(name = "MatchResult", module = "audio_pattern_detector", frozen, get_all)]
struct PyMatchResult {
    /// Detection times in seconds per clip name, sorted. Every clip of the
    /// detector has an entry (in pattern file order), empty when it was not found.
    detections: Py<PyDict>,
    /// Seconds of audio processed.
    duration_seconds: f64,
}

#[pymethods]
impl PyMatchResult {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "MatchResult(detections={}, duration_seconds={})",
            self.detections.bind(py).repr()?,
            self.duration_seconds.into_pyobject(py)?.repr()?
        ))
    }
}

/// Pattern clips prepared for matching; reusable across audio sources.
#[pyclass(name = "Detector", module = "audio_pattern_detector", frozen)]
struct PyDetector {
    detector: AudioPatternDetector,
}

impl PyDetector {
    /// Run `run` with the GIL released, forwarding detections to `on_detected`.
    fn run_match<F>(&self, py: Python<'_>, on_detected: Option<Py<PyAny>>, run: F) -> PyResult<PyMatchResult>
    where
        F: FnOnce(&AudioPatternDetector, Option<PatternDetectedCallback>, &PendingError) -> Result<(Option<PeakTimes>, f64)>
            + Send,
    {
        let pending = PendingError::new(None);
        let detector = &self.detector;
        let result = py.detach(|| {
            let mut notify = |clip_name: &str, timestamp: f64| {
                let Some(on_detected) = &on_detected else { return };
                if pending.lock().unwrap().is_some() {
                    return;
                }
                if let Err(error) = Python::attach(|py| on_detected.call1(py, (clip_name, timestamp))) {
                    store_error(&pending, error);
                }
            };
            run(detector, Some(&mut notify), &pending)
        });

        if let Some(error) = pending.into_inner().unwrap() {
            return Err(error);
        }
        let (peak_times, duration_seconds) = result.map_err(to_py_err)?;
        let mut peak_times = peak_times.unwrap_or_default();
        let detections = PyDict::new(py);
        for name in self.clip_names() {
            let mut times = peak_times.remove(&name).unwrap_or_default();
            times.sort_by(f64::total_cmp);
            detections.set_item(name, times)?;
        }
        Ok(PyMatchResult { detections: detections.unbind(), duration_seconds })
    }
}

#[pymethods]
impl PyDetector {
    #[new]
    #[pyo3(signature = (
        pattern_files,
        *,
        seconds_per_chunk = Some(DEFAULT_SECONDS_PER_CHUNK),
        target_sample_rate = DEFAULT_TARGET_SAMPLE_RATE,
        height_min = None,
    ))]
    fn new(
        py: Python<'_>,
        pattern_files: Vec<PathBuf>,
        seconds_per_chunk: Option<u32>,
        target_sample_rate: u32,
        height_min: Option<f32>,
    ) -> PyResult<Self> {
        let options = DetectorOptions {
            seconds_per_chunk,
            target_sample_rate,
            height_min,
        };
        let detector = py.detach(|| {
            let clips = load_pattern_clips(&pattern_files, target_sample_rate)?;
            AudioPatternDetector::new(clips, options)
        });
        Ok(Self { detector: detector.map_err(to_py_err)? })
    }

    /// Clip names in the order the pattern files were given.
    #[getter]
    fn clip_names(&self) -> Vec<String> {
        self.detector.get_config().clips.into_iter().map(|(name, _)| name).collect()
    }

    #[getter]
    fn seconds_per_chunk(&self) -> u32 {
        self.detector.seconds_per_chunk()
    }

    #[getter]
    fn target_sample_rate(&self) -> u32 {
        self.detector.target_sample_rate()
    }

    /// Computed configuration, the same data as the CLI's `show-config`.
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let config = self.detector.get_config();
        let clips = PyDict::new(py);
        for (name, clip) in &config.clips {
            let entry = PyDict::new(py);
            entry.set_item("duration_seconds", clip.duration_seconds)?;
            entry.set_item("sliding_window_seconds", clip.sliding_window_seconds)?;
            clips.set_item(name, entry)?;
        }
        let out = PyDict::new(py);
        out.set_item("default_seconds_per_chunk", config.default_seconds_per_chunk)?;
        out.set_item("min_chunk_size_seconds", config.min_chunk_size_seconds)?;
        out.set_item("sample_rate", config.sample_rate)?;
        out.set_item("clips", clips)?;
        Ok(out)
    }

    /// Match an audio file. WAV is decoded natively, anything else through ffmpeg.
    #[pyo3(signature = (audio_file, *, on_detected = None))]
    fn match_file(
        &self,
        py: Python<'_>,
        audio_file: PathBuf,
        on_detected: Option<Py<PyAny>>,
    ) -> PyResult<PyMatchResult> {
        self.run_match(py, on_detected, move |detector, callback, _| {
            find_clips_in_file(detector, &audio_file, callback, true)
        })
    }

    /// Match a WAV stream (mono, at the detector's sample rate) read from a
    /// binary file-like object until EOF, e.g. the stdout of an ffmpeg process.
    #[pyo3(signature = (stream, *, on_detected = None))]
    fn match_wav_stream(
        &self,
        py: Python<'_>,
        stream: Py<PyAny>,
        on_detected: Option<Py<PyAny>>,
    ) -> PyResult<PyMatchResult> {
        self.run_match(py, on_detected, move |detector, callback, pending| {
            let reader = PyReader { stream: &stream, pending };
            find_clips_in_wav_stream(detector, reader, callback, true)
        })
    }
}

/// Clip name the detector reports for a pattern file: the file name without
/// `.apd.toml` for pattern configs, otherwise without its last extension.
#[pyfunction]
fn clip_name(pattern_file: PathBuf) -> String {
    AudioClip::name_for_path(pattern_file)
}

#[pymodule]
#[pyo3(name = "audio_pattern_detector")]
fn audio_pattern_detector(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("APD_EXTENSION", APD_EXTENSION)?;
    module.add("DEFAULT_TARGET_SAMPLE_RATE", DEFAULT_TARGET_SAMPLE_RATE)?;
    module.add("DEFAULT_SECONDS_PER_CHUNK", DEFAULT_SECONDS_PER_CHUNK)?;
    module.add_class::<PyDetector>()?;
    module.add_class::<PyMatchResult>()?;
    module.add_function(wrap_pyfunction!(clip_name, module)?)?;
    Ok(())
}
