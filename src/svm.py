import numpy as np


class SVM:
    def __init__(
        self, learning_rate: float = 0.001, C: float = 1.0, n_iters: int = 1000
    ):
        self.lr = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.weights: np.ndarray = None
        self.bias = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)

        # for _ in range(self.n_iters):
        #     for idx, pt in enumerate(X):
        #         condition = y[idx] * (np.dot(pt, self.weights) + self.bias) >= 1
        #         if condition:
        #             self.weights -= self.lr * 2 * self.weights
        #         else:
        #             self.weights -= self.lr * (2 * self.weights - self.C * y[idx] * pt)
        #             self.bias -= self.lr * (-self.C * y[idx])
        for _ in range(self.n_iters):
            margin = y * (np.dot(X, self.weights) + self.bias)
            misclassified = margin < 1

            dw = 2 * self.weights - self.C * np.dot(
                X[misclassified].T, y[misclassified]
            )
            db = -self.C * np.sum(y[misclassified])

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_out = np.dot(X, self.weights) + self.bias
        return np.where(linear_out >= 0, 1, -1)
