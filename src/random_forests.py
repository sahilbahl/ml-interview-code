from typing import Optional

import numpy as np
from scipy.stats import mode

from src.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        num_data = X.shape[0]
        self.trees = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state,
            )
            for _ in range(self.n_estimators)
        ]
        for tree in self.trees:
            random_indicies = np.random.choice(num_data, num_data, replace=True)
            tree.fit(X[random_indicies], y[random_indicies])

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_prediction, _ = mode(predictions, axis=0)
        return final_prediction.flatten()
