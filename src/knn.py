import numpy as np

class KNN:

    def __init__(self, k: int):
        self.k = k
    
    def fit(self, X, y):
        self.train_X = X
        self.train_y = y
    
    def predict(self, X):
        predictions = []
        for data in X:
            distances = self._compute_distances(data)
            top_neighbour_idx =  np.argpartition(distances, self.k)[:self.k]

    def _compute_distances(self, data):
        distances = []

        for pt in self.train_X:
            distance = np.linalg.norm(pt - data)
            distances.append(distance)
        
        return distances
    
