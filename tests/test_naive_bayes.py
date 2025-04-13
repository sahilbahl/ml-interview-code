import numpy as np

from src.naive_bays import GaussianNaiveBayes


def test_fit():

    X = np.array([[1.0, 2.0], [1.1, 1.9], [5.0, 5.1], [5.2, 4.9]])
    y = np.array([0, 0, 1, 1])

    nb = GaussianNaiveBayes()  # your class
    nb.fit(X, y)

    # Check class priors
    assert np.allclose(nb.class_priors[0], 0.5)
    assert np.allclose(nb.class_priors[1], 0.5)

    # Check means
    expected_means_0 = np.mean(X[y == 0], axis=0)
    expected_means_1 = np.mean(X[y == 1], axis=0)
    assert np.allclose(nb.mean[0], expected_means_0)
    assert np.allclose(nb.mean[1], expected_means_1)

    # Check variances
    expected_vars_0 = np.var(X[y == 0], axis=0)
    expected_vars_1 = np.var(X[y == 1], axis=0)
    assert np.allclose(nb.variance[0], expected_vars_0)
    assert np.allclose(nb.variance[1], expected_vars_1)


def test_predict():
    # Training data: two clearly separated clusters
    X_train = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 5.0], [5.1, 5.1]])
    y_train = np.array([0, 0, 1, 1])

    # New samples (closer to one of the clusters)
    X_test = np.array(
        [[1.2, 2.0], [5.2, 4.9], [1.3, 2.1]]  # closer to class 0
    )  # closer to class 1

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    expected = np.array([0, 1, 0])
    assert np.array_equal(
        predictions, expected
    ), f"Expected {expected}, got {predictions}"


def test_predict_proba():
    # Training data
    X_train = np.array([[1.0, 2.0], [1.2, 2.1], [5.0, 6.0], [5.2, 5.9]])
    y_train = np.array([0, 0, 1, 1])

    # Test data
    X_test = np.array([[1.1, 2.0], [5.1, 6.0]])

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    # Predict class probabilities
    probas = model.predict_proba(X_test)

    # Check shape
    assert probas.shape == (2, 2)

    # Probabilities should be between 0 and 1
    assert np.all(probas >= 0) and np.all(probas <= 1)

    # Each row should sum to 1
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(2), rtol=1e-5)

    # Most probable class should match predict
    preds = model.predict(X_test)
    assert np.all(preds == np.argmax(probas, axis=1))
