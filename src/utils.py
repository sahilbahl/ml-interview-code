import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-z))
