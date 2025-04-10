from typing import Union

import numpy as np


# Union[float, np.ndarray] means the function can accept either a float or a numpy array
def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 1 / (1 + np.exp(-z))
