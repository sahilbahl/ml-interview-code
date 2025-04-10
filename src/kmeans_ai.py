from typing import Optional

import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        max_iters: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """
        KMeans clustering algorithm implementation

        Parameters:
        -----------
        n_clusters : int, default=8
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

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Compute k-means clustering

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self : object
            Fitted estimator
        """
        X = np.array(X)
        n_samples = X.shape[0]

        if self.n_clusters > n_samples:
            raise ValueError(
                "Number of clusters cannot be greater than number of data points."
            )

        # Initialise centroid by choosing k points at random from the dataset itself
        if self.random_state is not None:
            np.random.seed(self.random_state)
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        # Store a copy of the initial centroids
        self.centroids_ = X[random_indices].copy()

        # Perform clustering iterations
        for i in range(self.max_iters):
            self.labels_ = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)

            # Check convergence
            if np.allclose(self.centroids_, new_centroids, rtol=self.tol):
                break

            self.centroids_ = new_centroids

        # Calculate inertia (sum of squared distances to nearest centroid)
        self.inertia_ = self._calculate_inertia(X)

        return self

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each data point to the nearest centroid"""
        if self.centroids_ is None:
            raise ValueError("Centroids not initialized. Call fit first.")

        distances = np.zeros((X.shape[0], self.n_clusters))

        # Calculate Euclidean distance between each point and each centroid
        for i, centroid in enumerate(self.centroids_):
            # Calculate squared Euclidean distance using broadcasting
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)

        # Return the index of the closest centroid for each data point
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray) -> np.ndarray:
        """Update centroids to be the mean of all points assigned to that cluster"""
        if self.centroids_ is None:
            raise ValueError("Centroids not initialized. Call fit first.")
        if self.labels_ is None:
            raise ValueError("Labels not assigned. Call _assign_clusters first.")

        # Calculate new centroids by taking the mean of all points in each cluster
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        # Manually update the centroids
        for cluster in range(self.n_clusters):
            # Find points belonging to this cluster
            mask = self.labels_ == cluster
            if np.any(mask):  # If there are points in this cluster
                cluster_points = X[mask]
                # Calculate mean of points in cluster
                new_centroids[cluster] = np.mean(cluster_points, axis=0)
            else:
                # If no points in cluster, keep the old centroid
                new_centroids[cluster] = self.centroids_[cluster]

        # Update the centroids attribute
        self.centroids_ = new_centroids

        return self.centroids_

    def _calculate_inertia(self, X: np.ndarray) -> float:
        """Calculate inertia (sum of squared distances to nearest centroid)"""
        if self.centroids_ is None:
            raise ValueError("Centroids not initialized. Call fit first.")

        distances = np.zeros((X.shape[0], self.n_clusters))

        # Calculate squared Euclidean distance between each point and each centroid
        for i, centroid in enumerate(self.centroids_):
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)

        # Get the minimum distance for each point
        min_distances = np.min(distances, axis=1)

        # Sum up all the minimum distances
        return float(np.sum(min_distances))

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
        return self.labels_
