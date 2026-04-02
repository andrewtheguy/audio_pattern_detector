import numpy as np
import numpy.typing as npt

def find_peaks(
    data: npt.NDArray[np.floating[object]],
    *,
    height: float | None = None,
    distance: int | None = None,
    prominence: float | None = None,
) -> tuple[npt.NDArray[np.int64], dict[str, npt.NDArray[np.float64]]]: ...
