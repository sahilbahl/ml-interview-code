import numpy as np
import pytest

from src.som import SOM


@pytest.fixture
def sample_2d_data():
    """Create a simple 2D dataset for SOM testing"""
    # Two clusters that are clearly separated
    X = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.3],
            [0.3, 0.2],
            [0.1, 0.3],
            [0.3, 0.1],  # Cluster 1
            [0.8, 0.8],
            [0.7, 0.9],
            [0.9, 0.7],
            [0.8, 0.9],
            [0.9, 0.8],  # Cluster 2
        ]
    )
    return X


@pytest.fixture
def sample_iris_data():
    """Load the iris dataset for more complex SOM testing"""
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


def test_som_initialization():
    """Test that SOM initializes with correct parameters"""
    # Test initialization with default parameters
    som = SOM(map_size=(10, 10), input_dim=2)
    assert som.map_size == (10, 10)
    assert som.input_dim == 2
    assert som.learning_rate == 0.1  # Updated to match Kohonen algorithm description
    assert som.random_state is None
    assert som.num_iterations == 100  # Default number of iterations

    # Test initialization with custom parameters
    som = SOM(
        map_size=(5, 8),
        input_dim=4,
        learning_rate=0.1,
        num_iterations=500,
        random_state=42,
    )
    assert som.map_size == (5, 8)
    assert som.input_dim == 4
    assert som.learning_rate == 0.1  # Should always be 0.1 as per algorithm
    assert som.num_iterations == 500
    assert som.random_state == 42


def test_weights_initialization(sample_2d_data):
    """Test the initialization of SOM weights"""
    # Initialize SOM with fixed random state for reproducibility
    som = SOM(map_size=(3, 3), input_dim=2, random_state=42)

    # Check that weights are initialized
    som._initialize_weights()

    # Shape of weights should be (map_size[0], map_size[1], input_dim)
    assert som.weights.shape == (3, 3, 2)

    # Test that weights are initialized as random values between 0 and 1
    assert np.all(som.weights >= 0)
    assert np.all(som.weights <= 1)


def test_find_bmu():
    """Test the Best Matching Unit (BMU) finding mechanism"""
    # Create a simple SOM with known weights
    som = SOM(map_size=(2, 2), input_dim=2)
    som.weights = np.array(
        [[[0.1, 0.1], [0.9, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]  # Row 0  # Row 1
    )

    # Test with sample inputs that should match specific BMUs
    assert som._find_bmu(np.array([0.0, 0.0])) == (0, 0)  # Closest to [0.1, 0.1]
    assert som._find_bmu(np.array([1.0, 1.0])) == (0, 1)  # Closest to [0.9, 0.9]
    assert som._find_bmu(np.array([0.0, 1.0])) == (1, 0)  # Closest to [0.1, 0.9]
    assert som._find_bmu(np.array([1.0, 0.0])) == (1, 1)  # Closest to [0.9, 0.1]


def test_neighborhood_radius():
    """Test the neighborhood radius calculation"""
    # Create SOM with specific map size
    som = SOM(map_size=(10, 8), input_dim=2, num_iterations=100)

    # Initial radius should be max(width, height) / 2
    expected_initial_radius = 10 / 2  # max(10, 8) / 2
    assert som.neighbourhood_radius == expected_initial_radius

    # Test radius decay
    # At iteration 0
    radius_at_0 = som._calculate_radius(0)
    assert radius_at_0 == expected_initial_radius

    # At iteration 50 (middle of training)
    radius_at_50 = som._calculate_radius(50)
    assert radius_at_50 < radius_at_0

    # At final iteration
    radius_at_end = som._calculate_radius(99)
    assert radius_at_end < radius_at_50


def test_learning_rate_decay():
    """Test the learning rate decay function"""
    som = SOM(map_size=(5, 5), input_dim=2, num_iterations=100)

    # Initial learning rate should be 0.1
    assert som.learning_rate == 0.1

    # Test learning rate decay
    # At iteration 0
    lr_at_0 = som._calculate_learning_rate(0)
    assert lr_at_0 == 0.1

    # At iteration 50 (middle of training)
    lr_at_50 = som._calculate_learning_rate(50)
    assert lr_at_50 < lr_at_0

    # At final iteration
    lr_at_end = som._calculate_learning_rate(99)
    assert lr_at_end < lr_at_50


def test_neighborhood_influence():
    """Test the neighborhood influence calculation"""
    som = SOM(map_size=(5, 5), input_dim=2)

    # The center of the neighborhood (BMU)
    bmu = (2, 2)
    sigma = 1.0  # Example sigma value

    # Influence should be 1.0 at the BMU itself
    influence_at_bmu = som._calculate_influence((2, 2), bmu, sigma)
    assert influence_at_bmu == pytest.approx(1.0)

    # Influence should decrease with distance
    influence_at_1_step = som._calculate_influence((1, 2), bmu, sigma)
    assert influence_at_1_step < 1.0

    influence_at_2_steps = som._calculate_influence((0, 2), bmu, sigma)
    assert influence_at_2_steps < influence_at_1_step

    # Influence should be symmetric
    assert som._calculate_influence((1, 2), bmu, sigma) == som._calculate_influence(
        (3, 2), bmu, sigma
    )
    assert som._calculate_influence((2, 1), bmu, sigma) == som._calculate_influence(
        (2, 3), bmu, sigma
    )


def test_update_weights():
    """Test the update of weights during training"""
    # Create a simple SOM with known weights
    som = SOM(map_size=(3, 3), input_dim=2)
    som.weights = np.ones((3, 3, 2)) * 0.5  # Initialize all weights to 0.5

    # Use a sample input that will pull weights toward it
    sample = np.array([1.0, 1.0])
    bmu = (1, 1)  # Center node is the BMU

    # Current neighborhood radius (sigma)
    sigma = 1.0
    learning_rate = 0.1

    # Update weights with current learning rate and sigma
    som._update_weights(sample, bmu, learning_rate, sigma)

    # Check that weights at BMU moved significantly toward the sample
    assert som.weights[1, 1, 0] > 0.5
    assert som.weights[1, 1, 1] > 0.5

    # Calculate expected weight at BMU
    # Formula: W(t+1) = W(t) + α_t * θ_t * (V(t) - W(t))
    # At BMU itself, influence θ_t should be 1.0
    expected_weight_at_bmu = 0.5 + 0.1 * 1.0 * (1.0 - 0.5)  # = 0.5 + 0.05 = 0.55
    assert som.weights[1, 1, 0] == pytest.approx(expected_weight_at_bmu)

    # Check that farther away weights moved less
    # Corner node (0,0) is farther from BMU than edge node (0,1)
    assert som.weights[0, 0, 0] > 0.5  # Should still increase
    assert som.weights[0, 0, 0] < som.weights[1, 1, 0]  # But less than BMU
    assert (
        som.weights[0, 1, 0] > som.weights[0, 0, 0]
    )  # Edge node closer to BMU than corner

    # Calculate expected influence at corner (0,0)
    # Distance from (0,0) to BMU (1,1) is sqrt(2)
    d_corner = np.sqrt(2)
    influence_corner = np.exp(-(d_corner**2) / (2 * sigma**2))

    # Expected weight change at corner
    expected_weight_corner = 0.5 + learning_rate * influence_corner * (1.0 - 0.5)
    assert som.weights[0, 0, 0] == pytest.approx(expected_weight_corner)

    # Calculate expected influence at edge (0,1)
    # Distance from (0,1) to BMU (1,1) is 1
    d_edge = 1.0
    influence_edge = np.exp(-(d_edge**2) / (2 * sigma**2))

    # Expected weight change at edge
    expected_weight_edge = 0.5 + learning_rate * influence_edge * (1.0 - 0.5)
    assert som.weights[0, 1, 0] == pytest.approx(expected_weight_edge)


def test_fit(sample_2d_data):
    """Test the fitting process of the SOM"""
    X = sample_2d_data

    # Create SOM with fixed random state for reproducibility
    som = SOM(map_size=(3, 3), input_dim=2, random_state=42, num_iterations=10)

    # Fit the SOM
    som.fit(X)

    # Check that weights have been trained and have correct shape
    assert som.weights is not None
    assert som.weights.shape == (3, 3, 2)


def test_transform(sample_2d_data):
    """Test the transform method that maps data to BMUs"""
    X = sample_2d_data

    # Create and fit SOM
    som = SOM(map_size=(3, 3), input_dim=2, random_state=42, num_iterations=10)
    som.fit(X)

    # Transform data to get BMU coordinates
    bmus = som.transform(X)

    # Check shape is correct (n_samples, 2) for x,y coordinates
    assert bmus.shape == (len(X), 2)

    # Check that all coordinates are valid (within grid boundaries)
    assert np.all(bmus[:, 0] >= 0) and np.all(bmus[:, 0] < 3)
    assert np.all(bmus[:, 1] >= 0) and np.all(bmus[:, 1] < 3)


def test_fit_transform(sample_2d_data):
    """Test that fit_transform returns the BMU coordinates after fitting"""
    X = sample_2d_data

    # Create SOM
    som = SOM(map_size=(3, 3), input_dim=2, random_state=42, num_iterations=10)

    # Fit and transform in one step
    bmus = som.fit_transform(X)

    # Check shape
    assert bmus.shape == (len(X), 2)

    # Check they match what we'd get from separate fit and transform
    som2 = SOM(map_size=(3, 3), input_dim=2, random_state=42, num_iterations=10)
    som2.fit(X)
    bmus2 = som2.transform(X)

    np.testing.assert_array_equal(bmus, bmus2)


def test_fit_with_iterations(sample_2d_data):
    """Test that SOM training respects the number of iterations parameter"""
    X = sample_2d_data

    # Create SOM with different iteration counts
    som1 = SOM(map_size=(3, 3), input_dim=2, random_state=42, num_iterations=10)
    som2 = SOM(map_size=(3, 3), input_dim=2, random_state=42, num_iterations=50)

    # Train both SOMs
    som1.fit(X)
    weights1 = som1.weights.copy()

    som2.fit(X)
    weights2 = som2.weights.copy()

    # The weights should be different due to different training lengths
    # This is just a rough check - it could fail in rare cases
    assert not np.array_equal(weights1, weights2)


def test_fit_with_iris_data(sample_iris_data):
    """Test SOM training on a more complex dataset (Iris)"""
    X, y = sample_iris_data

    # Create and fit SOM
    som = SOM(map_size=(5, 5), input_dim=4, random_state=42, num_iterations=50)
    som.fit(X)

    # Check that weights have been trained
    assert som.weights is not None
    assert som.weights.shape == (5, 5, 4)

    # Get BMUs for all samples
    bmus = som.transform(X)

    # Ensure the BMU shape is correct
    assert bmus.shape == (X.shape[0], 2)


def test_bmu_convergence():
    # Create a dataset with clearly separable samples
    X = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])

    # Initialize SOM with a known random state for reproducibility
    som = SOM(map_size=(4, 4), input_dim=2, random_state=123, num_iterations=100)

    # Train the SOM over many iterations
    som.fit(X)

    # Get BMUs for all input samples
    bmus = som.transform(X)

    # Assert that the BMUs are distinct enough that similar inputs share BMUs
    # and distinct inputs do not. Here you could add your logic: for instance,
    # inputs with similar values should map close together.
    assert np.linalg.norm(bmus[0] - bmus[1]) > 1
    assert np.linalg.norm(bmus[2] - bmus[3]) > 1
