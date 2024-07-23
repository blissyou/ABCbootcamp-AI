import keras
import tensorflow as tf


model = tf.keras.models.Sequential(name = "XOR")
model.add(tf.keras.layers.Dense(units =2, input_shape = (2,), activation= 'sigmoid', name='INPUT'))
model.add(tf.keras.layers.Dense(units =1, activation= 'sigmoid', name='OUTPUT'))
model.compile(loss='mse', optimizer= 'sgd')

X = tf.constant([[0,0],[0,1],[1,0],[1,1]])
y= tf.constant([0,1,1,0])
print(model.summary())
model.fit(X,y,batch_size = 1,epochs=1000)
print(model.predict(X))
