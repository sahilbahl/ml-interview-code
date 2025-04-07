import pytest
import numpy as np
from src.kmeans import KMeans

@pytest.fixture
def simple_data():
    """Create a simple dataset with 3 clearly separated clusters."""
    # Create three distinct clusters
    cluster1 = np.random.rand(100, 2) + np.array([0, 0])
    cluster2 = np.random.rand(100, 2) + np.array([10, 0])
    cluster3 = np.random.rand(100, 2) + np.array([0, 10])
    X = np.vstack([cluster1, cluster2, cluster3])
    # True labels
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)
    return X, y


def test_kmeans_initialization():
    """Test KMeans initialization with different parameters."""
    # Default initialization
    kmeans = KMeans()
    assert kmeans.n_clusters == 3
    assert kmeans.max_iterations == 100
    assert kmeans.tol == 0.0001
    assert kmeans.random_state is None
    
    # Custom parameters
    kmeans = KMeans(n_clusters=5, max_iterations=200, tol=0.001, random_state=42)
    assert kmeans.n_clusters == 5
    assert kmeans.max_iterations == 200
    assert kmeans.tol == 0.001
    assert kmeans.random_state == 42


def test_kmeans_fit(simple_data):
    """Test KMeans fit method."""
    X, _ = simple_data
    kmeans = KMeans(n_clusters=3, random_state=42)
    
    # Test that fit returns self
    result = kmeans.fit(X)
    assert result is kmeans
    
    # Test that centroids, labels and inertia are set after fit
    assert kmeans.centroids is not None
    assert kmeans.labels_ is not None
    assert kmeans.inertia_ is not None
    
    # Check dimensions of results
    assert kmeans.centroids.shape == (3, 2)
    assert len(kmeans.labels_) == len(X)


def test_kmeans_predict(simple_data):
    """Test KMeans predict method."""
    X, _ = simple_data
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Get predictions
    predictions = kmeans.predict(X[:10])
    assert len(predictions) == 10
    
    # Test that each prediction is a valid cluster index
    assert all(0 <= p < 3 for p in predictions)


def test_kmeans_fit_predict(simple_data):
    """Test KMeans fit_predict method."""
    X, _ = simple_data
    kmeans = KMeans(n_clusters=3, random_state=42)
    
    # Get cluster assignments
    labels = kmeans.fit_predict(X)
    
    # Test that the method returns the same as labels_
    assert np.array_equal(labels, kmeans.labels_)
    assert len(labels) == len(X)


def test_predict_without_fit():
    """Test that predict raises error if called before fit."""
    kmeans = KMeans()
    X = np.random.rand(10, 2)
    
    with pytest.raises(ValueError):
        kmeans.predict(X)


def test_empty_input():
    """Test behavior with empty input."""
    kmeans = KMeans()
    X = np.array([]).reshape(0, 2)
    
    with pytest.raises(Exception):
        kmeans.fit(X)


def test_single_datapoint():
    """Test behavior with a single data point."""
    X = np.array([[1, 2]])
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(X)
    
    assert np.array_equal(kmeans.centroids, X)
    assert kmeans.labels_[0] == 0


def test_more_clusters_than_points():
    """Test behavior when k > n."""
    X = np.array([[1, 2], [3, 4]])
    kmeans = KMeans(n_clusters=3, random_state=42)
    
    with pytest.raises(Exception):
        kmeans.fit(X)


def test_inertia_calculation(simple_data):
    """Test that inertia is calculated correctly."""
    X, _ = simple_data
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Inertia should be positive
    assert kmeans.inertia_ > 0
    
    # Test with perfect clustering (each point is a centroid)
    X_small = np.array([[0, 0], [10, 0], [0, 10]])
    kmeans_perfect = KMeans(n_clusters=3, random_state=42)
    kmeans_perfect.fit(X_small)
    assert kmeans_perfect.inertia_ == 0