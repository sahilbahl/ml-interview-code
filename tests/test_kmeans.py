import numpy as np
import pytest
from kmeans_ai import KMeans

def test_kmeans_initialization():
    """Test the initialization of KMeans with different parameters"""
    kmeans = KMeans(n_clusters=3, max_iters=100, tol=1e-5, random_state=42)
    assert kmeans.n_clusters == 3
    assert kmeans.max_iters == 100
    assert kmeans.tol == 1e-5
    assert kmeans.random_state == 42
    assert kmeans.centroids is None
    assert kmeans.labels_ is None
    assert kmeans.inertia_ is None

def test_fit_when_k_greater_than_num_points_throws_error():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    kmeans = KMeans(n_clusters=5)
    with pytest.raises(ValueError, match="Number of clusters cannot be greater than number of data points."):
        kmeans.fit(X)

def test_centroid_initialization():
    """Test that centroids are initialized correctly from data points"""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    # Check that centroids were initialized
    assert kmeans.centroids is not None
    assert kmeans.centroids.shape == (2, 2)
    
    # With random_state=42, we should get deterministic initialization
    # Check the centroids are actual data points from X
    assert any(np.array_equal(kmeans.centroids[0], x) for x in X)
    assert any(np.array_equal(kmeans.centroids[1], x) for x in X)

def test_assign_clusters():
    """Test that _assign_clusters correctly assigns data points to the nearest centroid"""
    X = np.array([[1, 1], [2, 1], [1, 2],  # Cluster 1
                 [10, 10], [11, 10], [10, 11]])  # Cluster 2
    
    # Create KMeans instance with fixed centroids
    kmeans = KMeans(n_clusters=2)
    kmeans.centroids = np.array([[1, 1], [10, 10]])  # Set centroids manually to cluster centers
    
    # Call _assign_clusters method directly
    labels = kmeans._assign_clusters(X)
    
    # First 3 points should be assigned to first centroid (label 0)
    assert np.all(labels[:3] == 0)
    
    # Last 3 points should be assigned to second centroid (label 1)
    assert np.all(labels[3:] == 1)

def test_fit_convergence():
    """Test that KMeans converges to a solution"""
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, max_iters=10, random_state=42)
    kmeans.fit(X)
    
    # Check that labels and centroids are set
    assert kmeans.labels_ is not None

    # Check that the number of labels matches the number of samples
    assert len(kmeans.labels_) == X.shape[0]
    
    # Check that inertia is computed
    assert kmeans.inertia_ is not None

def test_update_centroids():
    """Test that _update_centroids correctly calculates new centroid positions"""
    X = np.array([[1, 1], [2, 1], [1, 2],    # Cluster 1
                 [10, 10], [11, 10], [10, 11]])  # Cluster 2
    
    # Create KMeans instance
    kmeans = KMeans(n_clusters=2)
    kmeans.centroids = np.array([[1, 1], [10, 10]])  # Set initial centroids
    
    # Assign points to clusters
    kmeans.labels_ = np.array([0, 0, 0, 1, 1, 1])  # First 3 points to cluster 0, last 3 to cluster 1
    
    # Update centroids
    kmeans._update_centroids(X)
    
    # Expected new centroids: mean of points in each cluster
    expected_centroids = np.array([
        [1.33333333, 1.33333333],  # Mean of first 3 points
        [10.33333333, 10.33333333]  # Mean of last 3 points
    ])
    
    # Check that centroids were updated correctly
    np.testing.assert_almost_equal(kmeans.centroids, expected_centroids, decimal=6)

def test_predict():
    """Test that predict assigns new data points to the nearest centroid"""
    # Train KMeans on some data
    X_train = np.array([[1, 1], [2, 1], [1, 2],  # Cluster 1
                       [10, 10], [11, 10], [10, 11]])  # Cluster 2
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    
    # Test data - should be classified into the two clusters
    X_test = np.array([[0, 0], [12, 12]])
    
    # Predict cluster for test data
    labels = kmeans.predict(X_test)
    
    # First point should be in the same cluster as the first training points
    # Second point should be in the same cluster as the last training points
    assert labels[0] == kmeans.labels_[0]  # Same cluster as points around (1,1)
    assert labels[1] == kmeans.labels_[3]  # Same cluster as points around (10,10)

def test_fit_predict():
    """Test that fit_predict returns the same labels as calling fit and then accessing labels_"""
    X = np.array([[1, 1], [2, 1], [1, 2],  # Cluster 1
                 [10, 10], [11, 10], [10, 11]])  # Cluster 2
    
    # Use fit followed by accessing labels_
    kmeans1 = KMeans(n_clusters=2, random_state=42)
    kmeans1.fit(X)
    labels1 = kmeans1.labels_
    
    # Use fit_predict directly
    kmeans2 = KMeans(n_clusters=2, random_state=42)
    labels2 = kmeans2.fit_predict(X)
    
    # Both methods should produce the same labels
    np.testing.assert_array_equal(labels1, labels2)