import numpy as np
import time
import matplotlib.pyplot as plt

SAMPLE_NUMBER = 10_000
np.random.seed(int(time.time()))
Xs = np.random.uniform(low = -2.0 ,high = 0.5,size = SAMPLE_NUMBER)

np.random.shuffle(Xs)
print(Xs[:10])
ys = (Xs+1.7)*(Xs+0.7)* (Xs-0.2)*(Xs-1.3)*(Xs-1.9)+0.2
plt.plot(Xs, ys,'r.')
plt.show()
ys += 0.1 * np.random.randn(SAMPLE_NUMBER)
plt.plot(Xs,ys,'b.')
plt.show()
np.save('../datas/noise.npy', ys)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = (train_test_split(Xs, ys, test_size = 0.2))
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

plt.plot(X_train,y_train,'b.',label = 'Train')
plt.plot(X_test,y_test,'r.',label = 'Test')
plt.legend()
plt.show()

import tensorflow as tf
model = tf.keras.Sequential([],name='Model')
Input_layer = tf.keras.Input(shape=(1,))
model.add(Input_layer)
model.add(tf.keras.layers.Dense(units=16, activation='relu',name='Layer1'))
model.add(tf.keras.layers.Dense(units=16, activation='relu',name='Layer2'))
model.add(tf.keras.layers.Dense(units =1 , name = 'OUTPUT'))

model.summary()
model.compile(optimizer='adam',loss='mse')
history = model.fit(X_train,y_train,epochs = 500)
print(history.history['loss'])
y_pred = model.predict(X_train)
print(f"최종 정확도 : {model.evaluate(X_train,y_pred)}")

plt.plot(history.history['loss'])
plt.show()
