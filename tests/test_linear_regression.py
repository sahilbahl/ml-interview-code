import numpy as np
import pytest
from src.linear_regression import LinearRegression  # Make sure to adjust the import based on the actual location

# Fixture to provide a simple dataset for testing
@pytest.fixture
def sample_data():
    # Example training data (2 features, 5 samples)
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 7]])  # Features are linearly independent
    y_train = np.array([5, 7, 9, 11, 13])
    
    return X_train, y_train

# Test case for the LinearRegression class
def test_linear_regression_fit(sample_data):
    X_train, y_train = sample_data
    
    # Initialize and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # We can assert the values of theta. The expected values are based on the normal equation.
    expected_theta = np.array([3, 2, 0])  # This is the expected theta from the known model: y = 1 + 2*x1 + 1*x2
    
    # Use np.testing.assert_allclose to compare the actual theta and expected theta
    np.testing.assert_allclose(model.theta, expected_theta, rtol=1e-5)
