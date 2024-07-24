import ssl

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

fashion_mnist = tf.keras.datasets.fashion_mnist  # 페션 관련 아이템
print(fashion_mnist)
print(fashion_mnist)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(X_train)
print(X_train.shape)
class_name = ['T-shirts','Trousers','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(visible=False)
    plt.xticks(ticks =list())
    plt.yticks(ticks =list())
    plt.imshow(X_train[i],cmap=plt.cm.binary)
plt.show()
(X_train,Xtest) =(np.divide(X_train,255.0),np.divide(X_test,255.0))
print(X_train)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu',name='LAYER1'),
    tf.keras.layers.Dense(units=64, activation='relu',name='LAYER2'),
    tf.keras.layers.Dense(units=16, activation='relu',name='LAYER3'),
    tf.keras.layers.Dense(units=10, activation='softmax',name='OUTPUT')
],name = 'MODEL')

model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
model.fit(x=X_train,y=y_train,epochs=100)
model.save('2024-07-24-1.keras')
