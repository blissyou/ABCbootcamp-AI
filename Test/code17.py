import ssl
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

fashion_mnist = tf.keras.datasets.fashion_mnist  # 페션 관련 아이템
print(fashion_mnist)
print(fashion_mnist)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train_CNN = X_train.reshape((-1,28,28,1))/255.0 # 데이터 갯수 백터 x y , 체널
X_test_CNN = X_test.reshape((-1,28,28,1))/255.0
model_CNN = tf.keras.Sequential([ # CNN
    tf.keras.layers.Conv2D(filters= 32,
                           kernel_size=(3,3),
                           padding='same',
                           activation='relu',
                           input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                 strides=2,padding='same'),
    tf.keras.layers.Conv2D(filters= 64,
                           kernel_size=(3,3),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                 strides=2,padding='same'),
    tf.keras.layers.Conv2D(filters= 32,
                           kernel_size=(3,3),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                 strides=2,padding='same'), # Convolution Network
    tf.keras.layers.Flatten(), #
    tf.keras.layers.Dense(units=128, activation='relu',name='LAYER1'),
    tf.keras.layers.Dense(units=64, activation='relu',name='LAYER2'),
    tf.keras.layers.Dense(units=16, activation='relu',name='LAYER3'),
    tf.keras.layers.Dense(units=10, activation='softmax',name='OUTPUT')
],name='CNN')

model_CNN.summary()
model_CNN.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model_CNN.fit(x=X_train_CNN,y=y_train,epochs=100)
# model_CNN.save('2024-07-24-CNN.keras')
model2 = tf.keras.models.load_model('/Users/choewonhyeong/Desktop/my_fucking_project/ABCbootcamp/AItest/Test/datas/2024-07-24-CNN.keras')
y_pred = model2.predict(X_test)

print(y_pred.shape)
print(f'y_pred[0]:{np.round(y_pred[0])}')
# print(f'y_test[0]:{np.round(y_test[0])}')
# print(f'y_pred[1]:{np.round(y_pred[1])}')
# print(f'y_test[1]:{np.round(y_test[1])}')
# print(f'y_pred[10]:{np.round(y_pred[10])}')
# print(f'y_test[10]:{np.round(y_test[10])}')
for i in range(0,10):
    print(f'y_pred{[i]}:{np.round(y_pred[i])}')
    print(f'y_test{[i]}:{np.round(y_test[i])}')
    time.sleep(2)
