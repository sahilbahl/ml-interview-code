import numpy as np
from src.linear_regression_gd import LinearRegressionGradientDescend
import pytest

# Fixture to provide a simple dataset for testing
@pytest.fixture
def sample_data_independent():
    # Example training data (2 features, 5 samples)
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 7]])  # Features are linearly independent
    y_train = np.array([5, 7, 9, 11, 13])
    
    return X_train, y_train

# Fixture to provide a simple dataset for testing
@pytest.fixture
def sample_data_linearly_dependent():
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([5, 7, 9, 11, 13])
    
    # Expected result: The true model is y = 1 + 2*x1 + 1*x2 (this is what we expect from the test)
    return X_train, y_train


# Fixture to provide a simple dataset for testing
@pytest.fixture
def sample_prediction():
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 7]])  # Features are linearly independent
    y_train = np.array([5, 7, 9, 11, 13])
    X_test = np.array([[10, 10], [-2, -5], [0, 2], [5, 9], [7, 0]])
    return X_train, y_train, X_test


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
    X_train, y_train = fixture
    
    # Initialize and fit the model
    model = LinearRegressionGradientDescend()
    model.fit(X_train, y_train)


def test_pred(sample_prediction):
    X_train, y_train, X_test =  sample_prediction

    lr = LinearRegressionGradientDescend(n_iters=100000)
    lr.fit(X_train, y_train)

    preds = lr.predict(X_test)

    print(preds)
