"""Math utilities."""

import numpy as np
import numpy.typing as npt


def softmax(arr: npt.NDArray[np.float_], tau: float):
    """Compute the softmax of an array.

    Parameters
    ----------
    arr : npt.NDArray[float]
        Array to compute the softmax of.
    tau : float
        Temperature parameter.

    Returns
    -------
    npt.NDArray[float]
        Softmax of `arr`.
    """
    arr = arr - np.max(arr, axis=-1, keepdims=True)
    exp = np.exp(arr / tau)
    return exp / np.sum(exp, axis=-1, keepdims=True)
