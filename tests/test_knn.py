from sklearn import datasets
from sklearn.model_selection import train_test_split

from src.knn import KNN


def test_knn():
    knn = KNN(3)
    iris_dataset = datasets.load_iris()
    X, y = iris_dataset.data, iris_dataset.target
    print(X)
    print(y)

    print(len(X))
    print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    error = y_test - y_pred
    print(error)
