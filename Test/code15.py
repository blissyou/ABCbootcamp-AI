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
# 모델 설정하기
from tensorflow.keras import Sequential,Input
model = tf.keras.models.Sequential([],name='MODEL')

# model.add(tf.keras.layers.Input(shape=(28,28),))

model.add(tf.keras.layers.Flatten(input_shape=(28,28),name='FLATTEN'))
model.add(tf.keras.layers.Dense(128,activation='relu',name='LAYER1'))
model.add(tf.keras.layers.Dense(64,activation='relu',name='LAYER2'))
model.add(tf.keras.layers.Dense(10,activation='softmax',name='OUTPUT'))

model.summary()
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.fit(X_train,y_train,epochs=100)
#
# model.evaluate(X_test,y_test)
# model.save('model.keras')

model2 = tf.keras.models.load_model('model.keras')
y_pred = model2.predict(X_test)

print(f'y_pred[0]:{y_pred[0].astype(int)}')
print(f'y_test[0]:{y_test[0]}')
print(f'y_pred[1]:{y_pred[1].astype(int)}')
print(f'y_test[1]:{y_test[1]}')

