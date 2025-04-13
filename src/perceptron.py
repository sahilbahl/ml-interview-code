import numpy as np


class Perceptron:
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _step_func(self, z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0, 1, 0).flatten()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        num_data, num_features = X.shape
        self.weights = np.random.rand(num_features, 1)
        self.bias = np.random.rand(1, 1)

        for iter in range(self.n_iters):
            for index, pt in enumerate(X):
                pred = self.predict(pt)

                self.weights += (
                    self.learning_rate * (y[index] - pred) * pt.reshape(-1, 1)
                )
                self.bias += self.learning_rate * (y[index] - pred)

        print(self.weights)
        print(self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._step_func(np.dot(X, self.weights) + self.bias)
