from collections import Counter

import numpy as np


class KNN:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self._compute_all_distances(X)
        # Get indices of k nearest neighbors
        neighbor_idxs = np.argpartition(distances, self.k, axis=1)[:, : self.k]
        # Fetch labels
        neighbor_labels = self.y_train[neighbor_idxs]
        # Mode across rows (axis=1)
        return np.array([Counter(row).most_common(1)[0][0] for row in neighbor_labels])

    def _compute_all_distances(self, X: np.ndarray) -> np.ndarray:
        # Efficient pairwise distance (Euclidean) using broadcasting:
        # (x - y)^2 = x^2 + y^2 - 2xy
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
        X_train_norm = np.sum(self.X_train**2, axis=1).reshape(1, -1)
        cross_term = np.dot(X, self.X_train.T)
        dists = np.sqrt(X_norm - 2 * cross_term + X_train_norm)
        return dists
