import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k: int):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for data in X:
            distances = self._compute_distances(data)
            nearest_neighbour_idx =  np.argpartition(distances, self.k)[:self.k]
            nearest_neighbour_preds = self.y_train[nearest_neighbour_idx]
            final_pred = Counter(nearest_neighbour_preds).most_common(1)[0][0]
            predictions.append(final_pred)
        return predictions

    def _compute_distances(self, data):
        distances = np.linalg.norm(self.X_train - data, axis=1)
        return distances