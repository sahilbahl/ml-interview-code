import numpy as np


def knn_search(reference: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """
    Find the indices of the k nearest neighbors in `reference` for each point in `query`.

    Parameters:
    - reference: np.ndarray of shape (N, D) representing N points in D dimensions.
    - query: np.ndarray of shape (M, D) representing M query points.
    - k: number of nearest neighbors to find.

    Returns:
    - neighbors: np.ndarray of shape (M, k), each row contains indices into `reference`
                 for the k nearest neighbors of the corresponding query point.
    """

    # Calculate euclidean distance based on the (a-b)^2 = a^2 + b^2 - 2a.b
    reference_squared = np.sum(reference**2, axis=1).reshape(1, -1)
    query_sqaured = np.sum(query**2, axis=1).reshape(-1, 1)
    distance = query_sqaured - 2 * np.dot(query, reference.T) + reference_squared
    top_k = np.argpartition(-distance, k, axis=1)[:, :k]

    return top_k


def knn_search_cosine(reference: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """
    Find the indices of the k nearest neighbors in `reference` for each point in `query`.

    Parameters:
    - reference: np.ndarray of shape (N, D) representing N points in D dimensions.
    - query: np.ndarray of shape (M, D) representing M query points.
    - k: number of nearest neighbors to find.

    Returns:
    - neighbors: np.ndarray of shape (M, k), each row contains indices into `reference`
                 for the k nearest neighbors of the corresponding query point.
    """
    dot_product = np.dot(query, reference.T)
    query_mag = np.linalg.norm(query, axis=1).reshape(-1, 1)
    reference_mag = np.linalg.norm(reference, axis=1).reshape(1, -1)
    distance = dot_product / (query_mag * reference_mag)
    top_k = np.argpartition(distance, k, axis=1)[:, :k]

    return top_k


reference = np.random.rand(1000, 128)
query = np.random.rand(10, 128)
k = 5

neighbors = knn_search_cosine(reference, query, k)
print(neighbors.shape)  # (10, 5)
