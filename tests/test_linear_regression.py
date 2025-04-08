import numpy as np
import pytest
from src.linear_regression import LinearRegression  # Make sure to adjust the import based on the actual location

# Fixture to provide a simple dataset for testing
@pytest.fixture
def sample_data_independent():
    # Example training data (2 features, 5 samples)
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 7]])  # Features are linearly independent
    y_train = np.array([5, 7, 9, 11, 13])
    expected_theta = np.array([3, 2, 0]) 
    
    return X_train, y_train, expected_theta

# Fixture to provide a simple dataset for testing
@pytest.fixture
def sample_data_linearly_dependent():
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([5, 7, 9, 11, 13])
    
    expected_theta = np.array([2, 1, 1]) 
    # Expected result: The true model is y = 1 + 2*x1 + 1*x2 (this is what we expect from the test)
    return X_train, y_train, expected_theta


@pytest.mark.parametrize(
    "data_fixture", [
        "sample_data_independent",
        "sample_data_linearly_dependent"
    ]
)
# Test case for the LinearRegression class
def test_linear_regression_fit(request, data_fixture):
     # Get the fixture using request
    fixture = request.getfixturevalue(data_fixture)
    X_train, y_train, expected_theta = fixture
    
    # Initialize and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Use np.testing.assert_allclose to compare the actual theta and expected theta
    np.testing.assert_allclose(model.theta, expected_theta, atol=0.1)
