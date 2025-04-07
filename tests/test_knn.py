from src.knn import KNN
from sklearn import datasets

def test_knn():
    knn = KNN(3)
    iris_dataset = datasets.load_iris()
    print(type(iris_dataset))

    knn.fit()