import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
print('Number of classes : ', format(len(np.unique(iris_Y))))
print('Number of data points: ', format(len(iris_Y)))

X0 = iris_X[iris_Y == 0, :]
X1 = iris_X[iris_Y == 1, :]
X2 = iris_X[iris_Y == 2, :]

X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=50)

clf = neighbors.KNeighborsClassifier(n_neighbors= 10, p = 2, weights= 'distance')
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print('Accuracy: ', format(100*accuracy_score(Y_test, Y_pred)))


