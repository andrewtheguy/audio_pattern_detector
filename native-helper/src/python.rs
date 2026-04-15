use crate::{
    resample_preserve_maxima_1d as rust_resample_preserve_maxima_1d,
    resample_envelope_1d as rust_resample_envelope_1d,
    find_peaks_1d as rust_find_peaks_1d, integrated_loudness as rust_integrated_loudness,
    loudness_normalize as rust_loudness_normalize,
    pearson_correlation_1d as rust_pearson_correlation_1d, resample_1d as rust_resample_1d,
    simpson_1d as rust_simpson_1d, FindPeaksOptions,
};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};

/// Borrow a contiguous 1-D float32 slice, converting from float64 only when needed.
fn with_f32_slice<R>(
    data: &Bound<'_, PyAny>,
    operation: impl FnOnce(&[f32]) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(arr) = data.cast::<PyArray1<f32>>() {
        let ro = arr.readonly();
        let slice = ro
            .as_slice()
            .map_err(|_| PyTypeError::new_err("data must be a contiguous 1D numpy array"))?;
        return operation(slice);
    }

    if let Ok(arr) = data.cast::<PyArray1<f64>>() {
        let ro = arr.readonly();
        let slice = ro
            .as_slice()
            .map_err(|_| PyTypeError::new_err("data must be a contiguous 1D numpy array"))?;
        let converted: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
        return operation(&converted);
    }

    Err(PyTypeError::new_err(
        "data must be a contiguous 1D numpy.ndarray with dtype float32 or float64",
    ))
}

/// Borrow a contiguous 1-D float64 slice, converting from float32 only when needed.
fn with_f64_slice<R>(
    data: &Bound<'_, PyAny>,
    operation: impl FnOnce(&[f64]) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(arr) = data.cast::<PyArray1<f64>>() {
        let ro = arr.readonly();
        let slice = ro
            .as_slice()
            .map_err(|_| PyTypeError::new_err("data must be a contiguous 1D numpy array"))?;
        return operation(slice);
    }

    if let Ok(arr) = data.cast::<PyArray1<f32>>() {
        let ro = arr.readonly();
        let slice = ro
            .as_slice()
            .map_err(|_| PyTypeError::new_err("data must be a contiguous 1D numpy array"))?;
        let converted: Vec<f64> = slice.iter().map(|&v| v as f64).collect();
        return operation(&converted);
    }

    Err(PyTypeError::new_err(
        "data must be a contiguous 1D numpy.ndarray with dtype float32 or float64",
    ))
}

fn with_two_f32_slices<R>(
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    operation: impl FnOnce(&[f32], &[f32]) -> PyResult<R>,
) -> PyResult<R> {
    with_f32_slice(x, |x_slice| {
        with_f32_slice(y, |y_slice| operation(x_slice, y_slice))
    })
}

#[pyfunction(
    name = "find_peaks",
    signature = (data, *, height=None, distance=None, prominence=None)
)]
fn find_peaks_py<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    height: Option<f32>,
    distance: Option<usize>,
    prominence: Option<f32>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyDict>)> {
    with_f32_slice(&data, |data_f32| {
        let options = FindPeaksOptions {
            height,
            distance,
            prominence,
        };
        let peaks = rust_find_peaks_1d(data_f32, &options);

        let peaks_i64: Vec<i64> = peaks.into_iter().map(|i| i as i64).collect();
        let peaks_array = peaks_i64.into_pyarray(py);
        let empty_dict = PyDict::new(py);

        Ok((peaks_array, empty_dict))
    })
}

#[pyfunction(name = "resample")]
fn resample_py<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    num_samples: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    with_f32_slice(&data, |data_f32| {
        let result = rust_resample_1d(data_f32, num_samples);
        Ok(result.into_pyarray(py))
    })
}

#[pyfunction(name = "resample_preserve_maxima")]
fn resample_preserve_maxima_py<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    num_samples: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if num_samples == 0 {
        return Err(PyValueError::new_err("num_samples must be greater than 0"));
    }

    with_f32_slice(&data, |data_f32| {
        let result = rust_resample_preserve_maxima_1d(data_f32, num_samples);
        if result.len() != num_samples {
            return Err(PyValueError::new_err(format!(
                "downsampled curve length {} not equal to num_samples {}",
                result.len(),
                num_samples
            )));
        }
        Ok(result.into_pyarray(py))
    })
}

#[pyfunction(name = "resample_envelope")]
fn resample_envelope_py<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    num_samples: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if num_samples == 0 {
        return Err(PyValueError::new_err("num_samples must be greater than 0"));
    }

    with_f32_slice(&data, |data_f32| {
        let result = rust_resample_envelope_1d(data_f32, num_samples);
        Ok(result.into_pyarray(py))
    })
}

#[pyfunction(name = "simpson")]
fn simpson_py(y: Bound<'_, PyAny>) -> PyResult<f64> {
    with_f64_slice(&y, |data_f64| Ok(rust_simpson_1d(data_f64)))
}

#[pyfunction(
    name = "integrated_loudness",
    signature = (data, sample_rate, block_size = 0.4)
)]
fn integrated_loudness_py(
    data: Bound<'_, PyAny>,
    sample_rate: u32,
    block_size: f64,
) -> PyResult<f64> {
    with_f32_slice(&data, |data_f32| {
        Ok(rust_integrated_loudness(data_f32, sample_rate, block_size))
    })
}

#[pyfunction(name = "loudness_normalize")]
fn loudness_normalize_py<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    current_lufs: f64,
    target_lufs: f64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    with_f32_slice(&data, |data_f32| {
        let result = rust_loudness_normalize(data_f32, current_lufs, target_lufs);
        Ok(result.into_pyarray(py))
    })
}

#[pyfunction(name = "pearson_correlation")]
fn pearson_correlation_py(x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<f64> {
    with_two_f32_slices(&x, &y, |x_f32, y_f32| {
        if x_f32.len() != y_f32.len() {
            return Err(PyValueError::new_err("arrays must have the same length"));
        }
        Ok(rust_pearson_correlation_1d(x_f32, y_f32))
    })
}

#[pymodule]
#[pyo3(name = "_native")]
fn native_helper(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "__all__",
        vec![
            "find_peaks",
            "resample",
            "resample_preserve_maxima",
            "resample_envelope",
            "simpson",
            "integrated_loudness",
            "loudness_normalize",
            "pearson_correlation",
        ],
    )?;
    module.add_function(wrap_pyfunction!(find_peaks_py, module)?)?;
    module.add_function(wrap_pyfunction!(resample_py, module)?)?;
    module.add_function(wrap_pyfunction!(resample_preserve_maxima_py, module)?)?;
    module.add_function(wrap_pyfunction!(resample_envelope_py, module)?)?;
    module.add_function(wrap_pyfunction!(simpson_py, module)?)?;
    module.add_function(wrap_pyfunction!(integrated_loudness_py, module)?)?;
    module.add_function(wrap_pyfunction!(loudness_normalize_py, module)?)?;
    module.add_function(wrap_pyfunction!(pearson_correlation_py, module)?)?;
    Ok(())
}
