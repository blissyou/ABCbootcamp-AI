from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
print(iris)
print(len(iris.get('target')))
print(len(iris.get('data')))

(X_train, X_test, y_train, y_test) = train_test_split(iris.data, iris.target, test_size=0.2,random_state=0)
print(len(X_train))
print(len(y_train))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(y_test)
