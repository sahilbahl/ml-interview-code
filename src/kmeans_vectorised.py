import numpy as np


def kmeans(
    X: np.ndarray, k: int, num_iters: int = 100, random_state: int = 42
) -> np.ndarray:
    """
    Run KMeans clustering on data X.

    Parameters:
    - X: np.ndarray of shape (N, D), where N is number of points, D is dimensions.
    - k: number of clusters
    - num_iters: max iterations
    - random_state: for reproducibility

    Returns:
    - centroids: np.ndarray of shape (k, D), final cluster centers
    """
    np.random.seed(random_state)
    N, D = X.shape

    # Step 1: Initialize centroids randomly from data
    indices = np.random.choice(N, k, replace=False)
    centroids = X[indices]

    X_squared = np.sum(X**2, axis=1).reshape(-1, 1)

    for _ in range(num_iters):
        centroids_squared = np.sum(centroids**2, axis=1).reshape(1, -1)
        distances = X_squared - 2 * np.dot(X, centroids.T) + centroids_squared
        labels = np.argmin(distances, axis=1)

        # Step 3: Update centroids
        for j in range(k):
            centroids[j] = X[labels == j].mean(axis=0)

    return centroids


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=10000, centers=5, n_features=10, random_state=0)
    centroids = kmeans(X, k=5, num_iters=10)
    print(centroids.shape)  # should be (5, 10)
