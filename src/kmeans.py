import numpy as np
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class KMeans:
    """
    Implementation of K-Means clustering algorithm using native Python.
    """
    
    def __init__(self, n_clusters: int = 3, max_iterations: int = 100, 
                 tol: float = 0.0001, random_state: Optional[int] = None):
        """
        Initialize KMeans instance.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters (K) to create.
        max_iterations : int, default=100
            Maximum number of iterations for the algorithm to converge.
        tol : float, default=0.0001
            Tolerance for considering the algorithm as converged.
        random_state : int, optional
            Seed for random number generation for reproducibility.
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Compute K-Means clustering.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        
        Returns:
        --------
        self : KMeans
            Fitted estimator.
        """
        # TODO: Implement the KMeans algorithm here
        # 1. Initialize centroids randomly
        # 2. Assign points to nearest centroid
        # 3. Update centroids based on assigned points
        # 4. Repeat steps 2-3 until convergence
        
        return self
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids randomly from the data points.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        
        Returns:
        --------
        centroids : array of shape (n_clusters, n_features)
            Initial centroids.
        """
        # TODO: Implement centroid initialization
        pass
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data points to assign.
        
        Returns:
        --------
        labels : array of shape (n_samples,)
            Cluster labels for each point.
        """
        # TODO: Implement cluster assignment
        pass
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids based on assigned points.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        labels : array-like of shape (n_samples,)
            Cluster labels for each point.
        
        Returns:
        --------
        centroids : array of shape (n_clusters, n_features)
            Updated centroids.
        """
        # TODO: Implement centroid update
        pass
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the sum of squared distances of samples to their closest centroid.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        labels : array of shape (n_samples,)
            Cluster labels for each point.
        
        Returns:
        --------
        inertia : float
            Sum of squared distances.
        """
        # TODO: Implement inertia calculation
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        
        Returns:
        --------
        labels : array of shape (n_samples,)
            Cluster labels for each point.
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet.")
        
        # TODO: Implement prediction logic
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        
        Returns:
        --------
        labels : array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_
    
    def plot_clusters(self, X: np.ndarray, title: str = "KMeans Clustering") -> None:
        """
        Plot the clusters and centroids (only works for 2D data).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data points to plot.
        title : str, default="KMeans Clustering"
        """
        if X.shape[1] != 2:
            print("Plotting is only supported for 2D data.")
            return
        
        # TODO: Implement visualization of clusters
        pass