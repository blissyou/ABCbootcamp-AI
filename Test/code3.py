import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
print(iris)
x = iris.data
y = iris.target
print(x.shape , y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.3)
regression = LogisticRegression()
# fit = 학습
regression.fit(X = X_train,y = y_train) # 120개의 훈련데이터와 120개의 정답을 가지고 훈련
regression.coef_ # weight 가중치
regression.intercept_ # bias Wx + b
print(f'가중치 : {regression.intercept_}')
print(f'바이어스: {regression.coef_}')
print(f"훈련하면서 맞춘 정답은 : {regression.score(X=X_train, y = y_train)}")
#print(regression.predict(X = x_test))
y_predicted = regression.predict(X= X_test)
print(f"내가 예상한 정답은 : { y_predicted}")
print(f"내가 가진 정답은    {y_test}")
from sklearn.metrics import accuracy_score
rounded_y_predicted = np.round(y_predicted).astype(int)
print(f"테스트한 모델의 정확도 : {accuracy_score(y_pred= rounded_y_predicted , y_true=y_test)}")

from matplotlib import pyplot as plt
plt.plot(rounded_y_predicted,y_test , color = 'red',linewidth = 4)
plt.show()
