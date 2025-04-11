from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class SOM:
    def __init__(
        self,
        map_size: tuple[int, int],
        input_dim: int,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        random_state: Optional[int] = None,
    ):
        self.map_size = map_size
        self.map_height, self.map_width = map_size

        self.input_dim = input_dim
        self.num_iterations = num_iterations

        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate

        self.initial_neighbourhood_radius = np.max(map_size) / 2
        self.neighbourhood_radius = self.initial_neighbourhood_radius

        self.time_const = self.num_iterations / np.log(self.neighbourhood_radius)

        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        self._initialize_weights()

    def _initialize_weights(self, X: np.ndarray = None) -> None:
        """
        Initialize the weights of the SOM
        """
        self.weights = None
        if X is not None:
            min_val_per_feature = np.min(X, axis=0).reshape(1, 1, -1)
            max_val_per_feature = np.max(X, axis=0).reshape(1, 1, -1)
            self.weights = np.random.uniform(
                min_val_per_feature,
                max_val_per_feature,
                (self.map_height, self.map_width, self.input_dim),
            )
        else:
            self.weights = np.random.rand(
                self.map_height, self.map_width, self.input_dim
            )

    def _find_bmu(self, x: np.ndarray) -> tuple[int, int]:
        distances = np.linalg.norm(x.reshape(1, 1, -1) - self.weights, axis=2)
        min_distance_node = np.unravel_index(distances.argmin(), distances.shape)
        return min_distance_node

    def _calculate_radius(self, iteration: int) -> float:
        """
        Calculate the neighbourhood radius
        """
        return self.initial_neighbourhood_radius * np.exp(-iteration / self.time_const)

    def _calculate_learning_rate(self, iteration: int) -> float:
        """
        Calculate the learning rate decay
        """
        return self.initial_learning_rate * np.exp(-iteration / self.time_const)

    def _calculate_influence(
        self, node: tuple[int, int], bmu: tuple[int, int], sigma: float
    ) -> float:
        distance = np.linalg.norm(np.array(node) - np.array(bmu))
        return np.exp(-1 * (distance**2) / (2 * sigma**2))

    def _update_weights(
        self,
        sample: np.ndarray,
        bmu: tuple[int, int],
        learning_rate: float,
        sigma: float,
    ) -> None:
        distances = np.linalg.norm(
            np.indices((self.map_height, self.map_width))
            - np.array(bmu).reshape(2, 1, 1),
            axis=0,
        )
        influence = np.exp(-1 * (distances**2) / (2 * sigma**2))
        change = learning_rate * influence
        self.weights += change.reshape(self.map_height, self.map_width, 1) * (
            sample - self.weights
        )

    def fit(self, X: np.ndarray) -> None:
        for iter in tqdm(range(1, self.num_iterations + 1)):
            radius = self._calculate_radius(iter)
            learning_rate = self._calculate_learning_rate(iter)

            for pt in X:
                bmu = self._find_bmu(pt)
                self._update_weights(pt, bmu, learning_rate, radius)

    def transform(self, X: np.ndarray) -> np.ndarray:
        bmus = np.zeros((X.shape[0], 2))
        for iter, pt in enumerate(X):
            bmu = np.array(self._find_bmu(pt)).reshape(1, -1)
            bmus[iter] = bmu
        return bmus

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    input_data = np.random.random((10, 3))
    som = SOM((100, 100), 3, 1000)
    som.fit(input_data)
    plt.imsave("1000_optimised.png", som.weights)
