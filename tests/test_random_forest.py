import numpy as np
import pytest
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.random_forests import RandomForestClassifier


@pytest.fixture
def iris_data():
    """Fixture to provide the iris dataset"""
    iris = datasets.load_iris()
    return iris.data, iris.target


@pytest.fixture
def binary_data():
    """Fixture to provide a binary classification dataset"""
    X, y = datasets.make_classification(
        n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42
    )
    return X, y


def test_random_forest_iris(iris_data):
    """Test RandomForestClassifier on iris dataset"""
    X, y = iris_data

    # Use only two classes to simplify
    mask = y < 2
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Our implementation
    rf = RandomForestClassifier(
        n_estimators=10, max_depth=5, min_samples_split=2, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Convert to numpy array if not already
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Ensure prediction shape matches test data
    assert len(y_pred) == len(y_test)

    # Check accuracy is reasonable (>70%)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.7, f"Accuracy is only {accuracy: .2f}"


def test_random_forest_binary(binary_data):
    """Test RandomForestClassifier on binary classification data"""
    X, y = binary_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Our implementation
    rf = RandomForestClassifier(
        n_estimators=15, max_depth=10, min_samples_split=2, random_state=42
    )
    rf.fit(X_train, y_train)
    our_pred = rf.predict(X_test)

    # Convert to numpy array if not already
    if not isinstance(our_pred, np.ndarray):
        our_pred = np.array(our_pred)

    # scikit-learn implementation for comparison
    sklearn_rf = SklearnRF(
        n_estimators=15, max_depth=10, min_samples_split=2, random_state=42
    )
    sklearn_rf.fit(X_train, y_train)
    sklearn_pred = sklearn_rf.predict(X_test)

    # Check accuracy of our implementation
    our_accuracy = accuracy_score(y_test, our_pred)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)

    # Our implementation should be reasonably accurate
    assert our_accuracy > 0.7, f"Our accuracy is only {our_accuracy: .2f}"

    # Our implementation should be within 15% of sklearn's accuracy
    assert abs(our_accuracy - sklearn_accuracy) < 0.15, (
        f"Our accuracy ({our_accuracy: .2f}) differs too much "
        f"from sklearn's ({sklearn_accuracy: .2f})"
    )


def test_random_forest_parameters():
    """Test RandomForestClassifier with different parameters"""
    X, y = datasets.make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=42
    )

    # Test with different n_estimators
    rf1 = RandomForestClassifier(n_estimators=5, random_state=42)
    rf1.fit(X, y)
    assert len(rf1.trees) == 5

    rf2 = RandomForestClassifier(n_estimators=10, random_state=42)
    rf2.fit(X, y)
    assert len(rf2.trees) == 10

    # Test with different max_depth
    rf_shallow = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    rf_deep = RandomForestClassifier(n_estimators=5, max_depth=10, random_state=42)

    rf_shallow.fit(X, y)
    rf_deep.fit(X, y)

    # Deeper trees should typically fit training data better
    shallow_pred = rf_shallow.predict(X)
    deep_pred = rf_deep.predict(X)

    shallow_acc = accuracy_score(y, shallow_pred)
    deep_acc = accuracy_score(y, deep_pred)

    # This might not always be true for all datasets, but is a reasonable expectation
    assert shallow_acc <= deep_acc, (
        f"Expected deeper trees to have higher training accuracy, "
        f"but got {shallow_acc: .2f} > {deep_acc: .2f}"
    )
