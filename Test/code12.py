import pandas as pd

df =pd.read_csv(filepath_or_buffer='datas/nonlinear.csv')
print(df.head())
import matplotlib.pyplot as plt

X = df.iloc[:,0]
print(X)
y = df.iloc[:,1]
print(y)
plt.scatter(X,y)
plt.show()

import tensorflow as tf
model = tf.keras.models.Sequential([], name="ML")
input_layer = tf.keras.layers.Input(shape=(1,))

dense_layer = tf.keras.layers.Dense(units= 6 ,activation='sigmoid',name='dense_layer')
dense_layer2 = tf.keras.layers.Dense(units= 5, activation='sigmoid',name='dense_layer2')
dense_layer3 = tf.keras.layers.Dense(units= 4, activation='sigmoid',name='dense_layer3')
dense_layer4 = tf.keras.layers.Dense(units= 1, activation='sigmoid',name='dense_layer4')

model.add(layer = input_layer)
model.add(dense_layer)
model.add(dense_layer2)
model.add(dense_layer3)
model.add(dense_layer4)
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.6)
              ,loss='mse')
print(type(X))
print(type(y))
X= X.to_numpy()
y= y.to_numpy()
print(X)
print(y)
print(X.shape)
print(y.shape)
tf_X = X.reshape(1_000,1)
print(tf_X)
print(tf_X.shape)
model.fit(x= tf_X,y=y,epochs=2_000)

model.save('weights_2024-07-23.h5')
predict_result = model.predict(x=X)
plt.scatter(X,y,color='blue')
plt.scatter(X,predict_result,color='red')
plt.show()
