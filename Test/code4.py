from scipy import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

(diabates_X , diabetes_y) = datasets.load_diabetes(return_X_y=True)

print(f"Data: {diabates_X}")
print(f"정답:{diabetes_y}")

diabetes_X_new = diabates_X[:, np.newaxis,2]
print(diabetes_X_new)
print(diabetes_X_new.shape)

X_train, X_test, y_train, y_test = train_test_split(diabetes_X_new, diabetes_y,
                                                    test_size=0.2, random_state=0)

regression:LinearRegression = LinearRegression()
regression.fit(X = X_train, y= y_train)
print(regression.score(X=X_train, y=y_train))
y_predicted =regression.predict(X = X_test)
print(y_predicted)
print(y_test)
plt.plot(y_predicted, y_test,'o',color='red')
plt.plot(y_predicted,y_test,color='blue',linewidth=3)
plt.show()
