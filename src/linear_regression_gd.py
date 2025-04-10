import numpy as np


class LinearRegressionGradientDescend:
    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> "LinearRegressionGradientDescend":
        self.X_train = X_train
        self.y_train = y_train.reshape(-1, 1)
        X = np.concatenate([np.ones((X_train.shape[0], 1)), self.X_train], axis=1)
        num_samples = self.X_train.shape[0]
        self.theta = np.random.random((X.shape[1], 1))

        for _ in range(self.n_iters):
            predictions = np.dot(X, self.theta)
            error = predictions - self.y_train
            self.theta = self.theta - self.lr * ((1 / num_samples)) * np.dot(X.T, error)

        print(self.theta)
        return self

    def predict(self, X_test: np.array) -> np.ndarray:
        X = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)
        return np.dot(X, self.theta)
