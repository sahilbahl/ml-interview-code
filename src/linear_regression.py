import numpy as np

class LinearRegression:
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, alpha: float = 0.01):
        self.X_train = X_train
        self.y_train = y_train

        X = np.concatenate([np.ones((X_train.shape[0], 1)), self.X_train], axis=1)
        X_transpose = X.T
        x_xt_mul = np.matmul(X_transpose, X)

        identity = np.eye(X.shape[1])
        identity[0, 0] = 0

        penalty_term = alpha * identity

        x_xt_mul = x_xt_mul + penalty_term
        x_xt_mul_inverse = np.linalg.inv(x_xt_mul)
        prod = np.matmul(x_xt_mul_inverse, X_transpose)
        self.theta = np.round(np.matmul(prod, y_train), 2)

        print("Theta values:", self.theta)  # Optionally print theta to verify

    def predict(self, X_test: np.array):
        X = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)
        pred = np.dot(X, self.theta)
        return pred
        


    