import numpy as np
import pytest

from src.svm import SVM


# 1. Linearly Separable Data
def test_linearly_separable():
    X = np.array([[2, 3], [1, 1], [2, 0], [7, 8], [8, 8], [9, 5]])
    y = np.array([-1, -1, -1, 1, 1, 1])

    svm = SVM(learning_rate=0.001, C=1000, n_iters=1000)
    svm.fit(X, y)

    preds = svm.predict(X)
    assert np.all(preds == y), f"Expected {y}, got {preds}"


# 2. XOR Data (not linearly separable)
def test_xor_case():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, -1])  # XOR pattern

    svm = SVM(learning_rate=0.001, C=1, n_iters=1000)
    svm.fit(X, y)

    preds = svm.predict(X)
    assert preds.shape == y.shape


# 3. Weights should be updated from initial state
def test_weights_updated():
    X = np.array([[1, 2], [2, 3], [3, 3], [4, 5]])
    y = np.array([1, 1, -1, -1])

    svm = SVM(learning_rate=0.001, C=10, n_iters=500)
    svm.fit(X, y)

    assert not np.allclose(svm.weights, 0), "Weights not updated"
    assert svm.bias != 0, "Bias not updated"


# 4. Predict before fit should raise error
def test_predict_before_fit():
    svm = SVM()
    X = np.array([[1, 2]])

    with pytest.raises(Exception):
        _ = svm.predict(X)


# 5. All one class — shouldn't crash
def test_all_one_class():
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([1, 1, 1])

    svm = SVM()
    svm.fit(X, y)

    preds = svm.predict(X)
    assert np.all(preds == 1)


# 6. Decision boundary test
def test_decision_boundary_sign():
    X = np.array([[1, 2], [2, 3], [3, 3], [4, 5]])
    y = np.array([-1, -1, 1, 1])

    svm = SVM(learning_rate=0.001, C=1.0, n_iters=1000)
    svm.fit(X, y)

    preds = svm.predict(X)
    assert set(preds).issubset({-1, 1})
