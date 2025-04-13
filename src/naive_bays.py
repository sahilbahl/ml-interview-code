import math

import numpy as np


class GaussianNaiveBayes:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Naive Bayes model to the training data.

        Parameters:
        - X: np.ndarray of shape (N, D), feature matrix
        - y: np.ndarray of shape (N,), target labels
        """
        num_data, num_feat = X.shape
        self.classes, indicies = np.unique(y, return_inverse=True)
        num_classes = len(self.classes)
        class_counts = np.bincount(indicies)

        self.mean = np.zeros((num_classes, num_feat))
        self.variance = np.zeros((num_classes, num_feat))
        class_feat_sum = np.zeros((num_classes, num_feat))
        class_feat_sum_squared = np.zeros((num_classes, num_feat))

        np.add.at(class_feat_sum, indicies, X)
        np.add.at(class_feat_sum_squared, indicies, X**2)

        self.mean = class_feat_sum / class_counts
        self.variance = class_feat_sum_squared / class_counts - self.mean**2

        self.class_priors = class_counts / num_data

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for input samples.

        Parameters:
        - X: np.ndarray of shape (N, D)

        Returns:
        - y_pred: np.ndarray of shape (N,)
        """
        num_pts = len(X)
        class_probabilities = np.zeros((num_pts, len(self.classes)))

        for c in range(len(self.classes)):
            post_xi_y = (1 / (np.sqrt(2 * math.pi * self.variance[c]))) * np.exp(
                -1 * (((X - self.mean[c]) ** 2) / (2 * self.variance[c]))
            )
            prob_y_x = np.log(self.class_priors[c]) + np.sum(np.log(post_xi_y), axis=1)
            class_probabilities[:, c] = prob_y_x
        y_pred = np.argmax(class_probabilities, axis=1)
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        Parameters:
        - X: np.ndarray of shape (N, D)

        Returns:
        - proba: np.ndarray of shape (N, C)
        """
        num_pts = len(X)
        class_probabilities = np.zeros((num_pts, len(self.classes)))

        for c in range(len(self.classes)):
            post_xi_y = (1 / (np.sqrt(2 * math.pi * self.variance[c]))) * np.exp(
                -1 * (((X - self.mean[c]) ** 2) / (2 * self.variance[c]))
            )
            prob_y_x = np.log(self.class_priors[c]) + np.sum(np.log(post_xi_y), axis=1)
            class_probabilities[:, c] = prob_y_x

        # Apply softmax to log probabilities for numerical stability
        max_log = np.max(class_probabilities, axis=1, keepdims=True)
        exp_probs = np.exp(class_probabilities - max_log)
        probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

        return probs
