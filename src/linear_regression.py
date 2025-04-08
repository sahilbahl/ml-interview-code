import numpy as np

class LinearRegression:
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

        X = np.concatenate([np.ones((X_train.shape[0], 1)), self.X_train], axis=1)
        X_transpose = X.T
        x_xt_mul = np.matmul(X_transpose, X)
        x_xt_mul_inverse = np.linalg.inv(x_xt_mul)
        prod = np.matmul(x_xt_mul_inverse, X_transpose)
        self.theta = np.round(np.matmul(prod, y_train), 2)

        print("Theta values:", self.theta)  # Optionally print theta to verify


    