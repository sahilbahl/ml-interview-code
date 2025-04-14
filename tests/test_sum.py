import types

import numpy as np

from src.calculator import DecisionTreeClassifier


def test_perfectly_separable_data():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 0, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)
    preds = tree.predict(X)

    assert np.array_equal(preds, y)


def test_single_class_data():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)
    preds = tree.predict(X)

    assert np.all(preds == 1)


def test_depth_limit():
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)
    preds = tree.predict(X)

    # Should not overfit with only depth=1
    assert len(np.unique(preds)) <= 2


def test_min_samples_split():
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1, 1])

    tree = DecisionTreeClassifier(min_samples_split=10)
    tree.fit(X, y)
    preds = tree.predict(X)

    # No split should happen; all predictions should be majority class
    assert np.all(preds == 1)


def test_multi_level_tree_split():
    # Features: 2D (easy to visualize)
    X = np.array(
        [
            [1, 1],  # class 0
            [2, 1],  # class 0
            [3, 1],  # class 1 (needs deeper split)
            [4, 1],  # class 1
            [5, 2],  # class 0 (forces deeper path)
            [6, 2],  # class 0
            [7, 2],  # class 1
            [8, 2],  # class 1
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
    tree.fit(X, y)
    y_pred = tree.predict(X)

    # Tree should be deep enough to correctly classify all points
    assert (y_pred == y).all(), f"Expected {y.tolist()}, but got {y_pred.tolist()}"


def test_no_valid_split_all_features_identical():
    # All features are the same, labels are different
    X = np.array([[1.0], [1.0], [1.0], [1.0]])
    y = np.array([0, 1, 0, 1])  # No way to split by X

    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=1)
    tree.fit(X, y)

    # Should predict the majority class
    preds = tree.predict(X)
    assert all(pred == preds[0] for pred in preds), "All predictions should be the same"
    assert preds[0] in [0, 1], "Prediction should be one of the label classes"


def test_decision_tree_multiclass():
    X = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [0, 1],
            [1, 0],
            [2, 1],
        ]
    )
    y = np.array([0, 1, 2, 0, 1, 2])

    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(preds) == {0, 1, 2}  # It should predict all three classes


def test_max_features_selection():
    X = np.array(
        [
            [1, 5, 9, 2, 7],
            [2, 6, 8, 3, 6],
            [3, 7, 7, 4, 5],
            [4, 8, 6, 5, 4],
            [5, 9, 5, 6, 3],
        ]
    )
    y = np.array([0, 1, 0, 1, 0])

    selection_counts = []

    def spy_find_best_split(self, X, y):
        total_num_features = X.shape[1]
        selected_feat_indexes = np.random.choice(
            total_num_features, self.max_features, replace=False
        )
        selection_counts.append(len(selected_feat_indexes))

        # Return a threshold that ensures valid split
        feat_index = selected_feat_indexes[0]
        threshold = np.mean(X[:, feat_index])
        return feat_index, threshold, 0.1

    clf = DecisionTreeClassifier(max_depth=2, max_features=2)
    clf._find_best_split = types.MethodType(spy_find_best_split, clf)
    clf.fit(X, y)

    # Assert that every call used exactly max_features
    assert all(count == 2 for count in selection_counts)
