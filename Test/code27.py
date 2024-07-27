import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import Sequential, Input
from numpy import shape

print(fdr.DataReader(symbol='005930', start='01/01/2016', end='12/23/2020'))
samsung = fdr.DataReader(symbol='005930', start='01/01/2016', end=None)
print(samsung)
print(samsung.columns)

open_values = samsung[['Open']]
print(open_values)
print(open_values.shape)
print(f"최솟값:{open_values.min()}")
print(f"최댓값:{open_values.max()}")

from sklearn.preprocessing import MinMaxScaler

scaled = MinMaxScaler(feature_range=(0, 1))  # 0~90300 -> (0,1)
scaled = scaled.fit_transform(open_values)
TEST_SIZE = 200
train_data = scaled[:-TEST_SIZE]
test_data = scaled[-TEST_SIZE:]


def make_sample(data, window):
    train = []
    target = []
    for i in range(len(data) - window):
        train.append(data[i:i + window])
        target.append(data[i + window])
    return np.array(train), np.array(target)


X_train, y_train = make_sample(train_data, 30)
print(f"X_train:{X_train}")
print(f"y_train:{X_train.shape}")
print(f"y_train:{y_train}")
print(f"y_train:{y_train.shape}")
# 훈련데이터 생성 : Tensor 3 인 데이털를 넣어줘야함

# LSTM(RNN 문제를 해결한 모델) 만들기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model = Sequential([],name='LSTM_MODEL')
model.add(Input(shape=(X_train.shape[1],1),name="INPUT"))
model.add(LSTM(units=32, activation ='tanh',return_sequences=True))
model.add(LSTM(units=16, activation ='tanh',return_sequences=False))
model.add(Dense(1,name="OUTPUT"))

import tensorflow as tf
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, epochs=100, batch_size=16,verbose=2)
# model.save('LSTM_MODEL.keras')
model2 = tf.keras.models.load_model('LSTM_MODEL.keras')
X_test,y_test = make_sample(test_data, 30)
print(prediction)
print(prediction.shape)


