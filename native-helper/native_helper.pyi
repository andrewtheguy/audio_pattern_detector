from typing import Any

import numpy as np
import numpy.typing as npt

def find_peaks(
    data: npt.NDArray[np.floating[Any]],
    *,
    height: float | None = None,
    distance: int | None = None,
    prominence: float | None = None,
) -> tuple[npt.NDArray[np.int64], dict[str, npt.NDArray[np.float64]]]: ...

def resample(
    data: npt.NDArray[np.floating[Any]],
    num_samples: int,
) -> npt.NDArray[np.float32]: ...

def simpson(y: npt.NDArray[np.floating[Any]]) -> float: ...

def integrated_loudness(
    data: npt.NDArray[np.floating[Any]],
    sample_rate: int,
    block_size: float = 0.4,
) -> float: ...

def loudness_normalize(
    data: npt.NDArray[np.floating[Any]],
    current_lufs: float,
    target_lufs: float,
) -> npt.NDArray[np.float32]: ...

def pearson_correlation(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
) -> float: ...
