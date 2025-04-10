import numpy as np
import pytest
from sklearn import datasets
from sklearn.decomposition import PCA as SKPCA
from sklearn.preprocessing import StandardScaler

from src.pca import PCA


@pytest.fixture
def simple_data():
    """Create a simple dataset for PCA testing"""
    # Simple 2D data with clear principal components
    X = np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ]
    )
    return X


@pytest.fixture
def iris_data():
    """Load the iris dataset for more complex PCA testing"""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


def test_pca_initialization():
    """Test that PCA initializes with correct parameters"""
    pca = PCA(n_components=2)
    assert pca.n_components == 2
    assert pca.components_ is None
    assert pca.explained_variance_ is None
    assert pca.explained_variance_ratio_ is None
    assert pca.mean_ is None

    # Test default initialization
    pca = PCA()
    assert pca.n_components is None


def test_pca_fit_simple(simple_data):
    """Test fitting PCA on a simple dataset"""
    X = simple_data

    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Create and fit PCA
    pca = PCA(n_components=2)
    pca.fit(X_std)

    # Check that components_ attribute exists and has correct shape
    assert pca.components_ is not None
    assert pca.components_.shape == (2, 2)

    # Check that mean_ attribute is correct
    assert pca.mean_ is not None

    np.testing.assert_allclose(pca.mean_, np.zeros((1, 2)), atol=1e-10)

    # Check that explained_variance_ and explained_variance_ratio_ attributes exist
    assert pca.explained_variance_ is not None
    assert pca.explained_variance_ratio_ is not None
    assert len(pca.explained_variance_) == 2
    assert len(pca.explained_variance_ratio_) == 2

    # Verify that the variance ratios sum to 1
    np.testing.assert_allclose(np.sum(pca.explained_variance_ratio_), 1.0, atol=1e-10)


def test_pca_transform_simple(simple_data):
    """Test transforming data with PCA"""
    X = simple_data

    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Create and fit PCA
    pca = PCA(n_components=2)
    pca.fit(X_std)

    # Transform the data
    X_transformed = pca.transform(X_std)

    # Check the shape of the transformed data
    assert X_transformed.shape == (X.shape[0], 2)

    # Verify that the transformed data has the expected properties:
    # The variance along first PC should be higher than along second PC
    assert np.var(X_transformed[:, 0]) >= np.var(X_transformed[:, 1])


def test_pca_fit_transform_simple(simple_data):
    """Test fit_transform method on simple data"""
    X = simple_data

    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Create PCA
    pca = PCA(n_components=2)

    # Fit and transform in one step
    X_transformed = pca.fit_transform(X_std)

    # Check the shape of the transformed data
    assert X_transformed.shape == (X.shape[0], 2)

    # Test that fit_transform equals fit followed by transform
    pca2 = PCA(n_components=2)
    pca2.fit(X_std)
    X_transformed2 = pca2.transform(X_std)

    np.testing.assert_allclose(X_transformed, X_transformed2, atol=1e-10)


def test_pca_dimension_reduction(simple_data):
    """Test PCA dimension reduction"""
    X = simple_data

    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Create PCA with reduced dimensions
    pca = PCA(n_components=1)
    X_reduced = pca.fit_transform(X_std)

    # Check the shape of the transformed data
    assert X_reduced.shape == (X.shape[0], 1)

    # Check explained variance is for one component only
    assert len(pca.explained_variance_) == 1
    assert len(pca.explained_variance_ratio_) == 1


def test_pca_with_iris_dataset(iris_data):
    """Test PCA on the Iris dataset"""
    X, y = iris_data

    # Standardize the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    # Verify shape
    assert X_pca.shape == (X.shape[0], 2)

    # Verify variance explanation is in descending order
    assert pca.explained_variance_[0] >= pca.explained_variance_[1]

    # Test against sklearn implementation
    sk_pca = SKPCA(n_components=2)
    X_sk_pca = sk_pca.fit_transform(X_std)

    # Compare explained variance ratios (may differ in sign but should be similar in magnitude)
    np.testing.assert_allclose(
        np.abs(pca.explained_variance_ratio_),
        np.abs(sk_pca.explained_variance_ratio_),
        atol=0.01,
    )


def test_pca_with_n_components_none(simple_data):
    """Test PCA when n_components is None (should retain all components)"""
    X = simple_data

    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Create PCA with None n_components
    pca = PCA()
    X_transformed = pca.fit_transform(X_std)

    # Check the shape of the transformed data
    assert X_transformed.shape == X_std.shape
    assert pca.components_.shape == (X.shape[1], X.shape[1])


def test_pca_transform_new_data(simple_data):
    """Test applying transform to new, unseen data"""
    X = simple_data

    # Standardize the training data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - mean) / std

    # Create and fit PCA
    pca = PCA(n_components=2)
    pca.fit(X_std)

    # Create new test data
    X_new = np.array([[2.0, 2.0], [1.0, 1.0]])

    # Standardize the test data using training mean and std
    X_new_std = (X_new - mean) / std

    # Transform the new data
    X_new_transformed = pca.transform(X_new_std)

    # Check the shape of the transformed data
    assert X_new_transformed.shape == (X_new.shape[0], 2)


def test_pca_inverse_transform(simple_data):
    """Test reconstructing data from principal components"""
    X = simple_data

    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Create and fit PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X_std)

    # Reconstruct the data
    X_reconstructed = pca.inverse_transform(X_transformed)

    # Check the shape of the reconstructed data
    assert X_reconstructed.shape == X_std.shape

    # With all components, reconstruction should be close to original
    np.testing.assert_allclose(X_std, X_reconstructed, atol=1e-10)

    # Test with reduced dimensions
    pca_reduced = PCA(n_components=1)
    X_reduced = pca_reduced.fit_transform(X_std)
    X_reconstructed_reduced = pca_reduced.inverse_transform(X_reduced)

    # Check the shape of the reconstructed data
    assert X_reconstructed_reduced.shape == X_std.shape

    # With fewer components, reconstruction won't be perfect
    # but the approximation error should be reasonable
    reconstruction_error = np.mean((X_std - X_reconstructed_reduced) ** 2)
    assert reconstruction_error < 0.5  # This threshold may need adjustment
