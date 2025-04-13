import numpy as np

from src.perceptron import Perceptron  # adjust this import to your file structure


def test_linearly_separable():
    # Simple linearly separable dataset (AND logic)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND logic

    model = Perceptron(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y)


def test_all_zeros():
    X = np.zeros((5, 2))
    y = np.zeros(5)

    model = Perceptron()
    model.fit(X, y)
    preds = model.predict(X)

    assert np.all(preds == 0)


def test_all_ones():
    X = np.ones((5, 2))
    y = np.ones(5)

    model = Perceptron()
    model.fit(X, y)
    preds = model.predict(X)

    assert np.all(preds == 1)


def test_misclassified_input():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])

    model = Perceptron(learning_rate=1.0, n_iters=5)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y)
