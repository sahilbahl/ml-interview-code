from typing import Optional

import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iters: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        """
        KMeans clustering algorithm implementation

        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters (K) to form
        max_iters : int, default=300
            Maximum number of iterations for a single run
        tol : float, default=1e-4
            Tolerance for stopping criterion
        random_state : int, default=None
            Seed for random initialization
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.distances: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "KMeans":
        num_samples, n_features = X.shape

        if self.n_clusters > num_samples:
            raise ValueError(
                "Number of clusters cannot be greater than number of data points."
            )

        initial_centroid_indices = np.random.choice(
            num_samples, self.n_clusters, replace=False
        )
        self.centroids_ = X[initial_centroid_indices]

        for _ in range(self.max_iters):
            self.labels_ = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)

            if np.allclose(self.centroids_, new_centroids, atol=self.tol):
                break

            self.centroids_ = new_centroids

        self.inertia_ = self._calculate_inertia(X)
        return self

    def _calculate_inertia(self, X: np.ndarray) -> float:
        """Calculate inertia (sum of squared distances to nearest centroid)"""
        distances = np.zeros((X.shape[0], self.n_clusters))

        # Calculate squared Euclidean distance between each point and each centroid
        assert (
            self.centroids_ is not None
        ), "Centroids must be initialized before calculating inertia"
        for i, centroid in enumerate(self.centroids_):  # centroid is of type np.ndarray
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)

        # Get the minimum distance for each point
        min_distances = np.min(distances, axis=1)

        # Sum up all the minimum distances
        return np.sum(min_distances)

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        self.distances = self._compute_distances(X)
        min_per_row = np.argmin(self.distances, axis=1)
        return min_per_row

    def _update_centroids(self, X: np.ndarray) -> np.ndarray:
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        assert (
            self.labels_ is not None
        ), "Labels must be initialized before updating centroids"
        assert (
            self.centroids_ is not None
        ), "Centroids must be initialized before updating centroids"

        for idx in range(self.n_clusters):
            cluster_pts = X[self.labels_ == idx]
            if len(cluster_pts) > 0:
                new_centroids[idx] = np.mean(cluster_pts, axis=0)
            else:
                new_centroids[idx] = self.centroids_[idx]

        return new_centroids

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        distances = np.zeros((X.shape[0], self.n_clusters))

        assert (
            self.centroids_ is not None
        ), "Centroids must be initialized before computing distances"

        for idx in range(self.n_clusters):
            centroid: np.ndarray = self.centroids_[idx].reshape(1, -1)
            distances[:, idx] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))

        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' before using 'predict'.")

        X = np.array(X)

        # Assign each data point to the nearest centroid
        return self._assign_clusters(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to transform

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        self.fit(X)
        assert self.labels_ is not None, "Labels must be initialized after fitting"
        return self.labels_
