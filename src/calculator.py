from typing import List
from typing import Union

import numpy as np


class Calculator:
    def add(
        self, a: Union[List, int, np.ndarray], b: Union[List, int, np.ndarray]
    ) -> Union[float, int, np.ndarray]:
        """Add two numbers, vectors, or matrices.
        This function can handle both scalar and vector/matrix addition.
        It will convert lists to numpy arrays if necessary.

        Args:
            a (Union[List, int, np.ndarray]): int, float, list, or numpy array
            b (Union[List, int, np.ndarray]): int, float, list, or numpy array

        Returns:
            np.ndarray: Sum of a and b as a numpy array.
        """
        a = np.array(a)
        b = np.array(b)
        return a + b


if __name__ == "__main__":
    # Example usage
    calc = Calculator()
    print(calc.add(1, 2))  # Output: 3
    print(calc.add([1, 2], [3, 4]))  # Output: [4 6]
    print(
        calc.add(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
    )  # Output: [[ 6  8]
    #          [10 12]]
