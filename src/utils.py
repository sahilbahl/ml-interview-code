import numpy as np
from typing import Tuple


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
    -----------
    x1 : array-like
        First point.
    x2 : array-like
        Second point.
    
    Returns:
    --------
    distance : float
        Euclidean distance between the points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def generate_random_data(n_samples: int = 300, n_features: int = 2, 
                        n_centers: int = 3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random data clusters for testing K-means algorithm.
    
    Parameters:
    -----------
    n_samples : int, default=300
        Number of samples to generate.
    n_features : int, default=2
        Number of features for each sample.
    n_centers : int, default=3
        Number of centers to generate.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        Generated samples.
    y : ndarray of shape (n_samples,)
        Ground truth labels.
    """
    # TODO: Implement data generation function
    pass