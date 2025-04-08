from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from src.logistic_regression import LogisticRegression


def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

def test_pred():
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    logisitic_reg = LogisticRegression()
    logisitic_reg.fit(X_train, y_train)

    y_pred = logisitic_reg.predict(X_test)
    acc = accuracy(y_pred, y_test)
    assert acc > 0.9