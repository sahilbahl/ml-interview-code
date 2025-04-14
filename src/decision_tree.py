from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Node:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    gini_split_val: Optional[float] = None
    left_child: Optional["Node"] = None
    right_child: Optional["Node"] = None
    value: Optional[int] = None


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 1,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def _gini_impurity(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        prob_sq_sum = np.sum((counts / len(y)) ** 2)
        return 1 - prob_sq_sum

    def _build_pure_node(self, y: np.ndarray) -> Node:
        value = Counter(y).most_common()[0][0]
        return Node(value=value)

    def _build_node(self, X: np.ndarray, y: np.ndarray, current_depth: int) -> Node:
        unique_classes = np.unique(y)

        # Pure node or min sample check
        if len(unique_classes) == 1 or len(X) < self.min_samples_split:
            return self._build_pure_node(y)

        best_feat_index, best_threshold, best_gini_split = self._find_best_split(X, y)

        if best_feat_index is None or current_depth == self.max_depth:
            return self._build_pure_node(y)
        else:
            node = Node(
                feature_index=best_feat_index,
                threshold=best_threshold,
                gini_split_val=best_gini_split,
            )
            left_split = np.where(X[:, best_feat_index] <= best_threshold)[0]
            right_split = np.where(X[:, best_feat_index] > best_threshold)[0]

            node.left_child = self._build_node(
                X[left_split], y[left_split], current_depth=current_depth + 1
            )
            node.right_child = self._build_node(
                X[right_split], y[right_split], current_depth=current_depth + 1
            )
            return node

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[Optional[int], Optional[float], Optional[float]]:
        total_num_features = X.shape[1]
        selected_feat_indexes = list(range(total_num_features))

        if self.max_features is not None:
            selected_feat_indexes = np.random.choice(
                total_num_features, self.max_features, replace=False
            )

        unique_values = {
            feat_index: np.unique(X[:, feat_index])
            for feat_index in selected_feat_indexes
        }
        num_data_pts = X.shape[0]

        best_feat_index = None
        best_threshold = None
        best_gini_split = None
        for feat_index in selected_feat_indexes:
            vals = unique_values[feat_index]

            thresholds = (vals[1:] + vals[:-1]) / 2
            for threshold in thresholds:
                left_indices = np.where(X[:, feat_index] <= threshold)[0]
                right_indices = np.where(X[:, feat_index] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_gini = self._gini_impurity(y[left_indices])
                right_gini = self._gini_impurity(y[right_indices])

                weighted_gini = (
                    len(left_indices) / num_data_pts * left_gini
                    + len(right_indices) / num_data_pts * right_gini
                )

                if best_gini_split is None or weighted_gini < best_gini_split:
                    best_feat_index = feat_index
                    best_threshold = threshold
                    best_gini_split = weighted_gini

        return best_feat_index, best_threshold, best_gini_split

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root_node = self._build_node(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(self.root_node, x) for x in X])

    def _traverse_tree(self, node: Node, x: np.ndarray) -> int:
        # Leaf node
        if node.value is not None:
            return node.value
        else:
            if x[node.feature_index] <= node.threshold:
                if node.left_child is None:
                    raise ValueError("Left child node is None")
                return self._traverse_tree(node.left_child, x)
            else:
                if node.right_child is None:
                    raise ValueError("Right child node is None")
                return self._traverse_tree(node.right_child, x)
