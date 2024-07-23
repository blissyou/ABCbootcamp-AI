# import pandas as pd
#
# data_frane = pd.read_csv(
#     '/Users/choewonhyeong/Desktop/my_fucking_project/ABCbootcamp/AItest/scores_em.csv', index_col='student number')
# print(data_frane.head())
# """
# 선형회귀 (linear) 공학자는 LTI(linear time invarment)
# y = f(x)  x -> 독립  y-> 종속
# f(x)가 y에 mapping 된다 라고 말한다.
#
# 미분
# 그래프 -> 기울기
#
# 비선형에서 기울기를 알고싶으면 순간미분을 하면된다.
#
# AI에서 머신러닝은 계속해서 값을 넣고 돌리면서 오차를 줄여나가는 방식이다.
#
#
# """
import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.0, 1.0, 2.0])
y = np.array([3.0, 3.5, 5.5])
W = 0  # 기울기
b = 0  # 절편
lrate = 0.01  # 학습률
epochs = 1000  # 반복 횟수
n = float(len(X))  # 입력 데이터의 개수
# 경사 하강법
for i in range(epochs):
    y_pred = W * X + b  # 예측값
    dW = (2 / n) * sum(X * (y_pred - y))
    db = (2 / n) * sum(y_pred - y)
    W = W - lrate * dW  # 기울기 수정
    b = b - lrate * db  # 절편 수정
# 기울기와 절편을 출력한다.
print(W, b)
# 예측값을 만든다.
y_pred = W * X + b
# 입력 데이터를 그래프 상에 찍는다.
plt.scatter(X, y)
# 예측값은 선그래프로 그린다.
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)],
         color='red')
plt.show()
# 결과
# 1.2532418085611319 ->기울기 2.745502230882486 -> 절편
