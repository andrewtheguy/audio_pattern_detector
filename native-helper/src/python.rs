use crate::{
    find_peaks_1d as rust_find_peaks_1d, resample_1d as rust_resample_1d,
    simpson_1d as rust_simpson_1d, FindPeaksOptions,
};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyDict};

/// Extract f32 data from a 1-D numpy array, converting from f64 if needed.
fn extract_f32_data<'py>(data: &Bound<'py, PyAny>) -> PyResult<Vec<f32>> {
    // Try f32 first.
    if let Ok(arr) = data.cast::<PyArray1<f32>>() {
        let ro = arr.readonly();
        return Ok(ro
            .as_slice()
            .map_err(|_| PyTypeError::new_err("data must be a contiguous 1D numpy array"))?
            .to_vec());
    }
    // Try f64 and convert.
    if let Ok(arr) = data.cast::<PyArray1<f64>>() {
        let ro = arr.readonly();
        let slice = ro
            .as_slice()
            .map_err(|_| PyTypeError::new_err("data must be a contiguous 1D numpy array"))?;
        return Ok(slice.iter().map(|&v| v as f32).collect());
    }
    Err(PyTypeError::new_err(
        "data must be a contiguous 1D numpy.ndarray with dtype float32 or float64",
    ))
}

/// Extract f64 data from a 1-D numpy array, converting from f32 if needed.
fn extract_f64_data<'py>(data: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    // Try f64 first.
    if let Ok(arr) = data.cast::<PyArray1<f64>>() {
        let ro = arr.readonly();
        return Ok(ro
            .as_slice()
            .map_err(|_| PyTypeError::new_err("data must be a contiguous 1D numpy array"))?
            .to_vec());
    }
    // Try f32 and convert.
    if let Ok(arr) = data.cast::<PyArray1<f32>>() {
        let ro = arr.readonly();
        let slice = ro
            .as_slice()
            .map_err(|_| PyTypeError::new_err("data must be a contiguous 1D numpy array"))?;
        return Ok(slice.iter().map(|&v| v as f64).collect());
    }
    Err(PyTypeError::new_err(
        "data must be a contiguous 1D numpy.ndarray with dtype float32 or float64",
    ))
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
    let data_f32 = extract_f32_data(&data)?;

    let options = FindPeaksOptions {
        height,
        distance,
        prominence,
    };
    let peaks = rust_find_peaks_1d(&data_f32, &options);

    let peaks_i64: Vec<i64> = peaks.into_iter().map(|i| i as i64).collect();
    let peaks_array = peaks_i64.into_pyarray(py);
    let empty_dict = PyDict::new(py);

    Ok((peaks_array, empty_dict))
}

#[pyfunction(name = "resample")]
fn resample_py<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    num_samples: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let data_f32 = extract_f32_data(&data)?;
    let result = rust_resample_1d(&data_f32, num_samples);
    Ok(result.into_pyarray(py))
}

#[pyfunction(name = "simpson")]
fn simpson_py(y: Bound<'_, PyAny>) -> PyResult<f64> {
    let data_f64 = extract_f64_data(&y)?;
    Ok(rust_simpson_1d(&data_f64))
}

#[pymodule]
#[pyo3(name = "native_helper")]
fn native_helper(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__all__", vec!["find_peaks", "resample", "simpson"])?;
    module.add_function(wrap_pyfunction!(find_peaks_py, module)?)?;
    module.add_function(wrap_pyfunction!(resample_py, module)?)?;
    module.add_function(wrap_pyfunction!(simpson_py, module)?)?;
    Ok(())
}
