import matplotlib.pyplot as plt
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data(path="/Users/choewonhyeong/Desktop/my_fucking_project/ABCbootcamp/AItest/Test"
                                           "/mnist.npz"))


print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
plt.imshow(train_images[1], cmap='BuPu')
plt.show()
print(train_labels[0])

model= tf.keras.models.Sequential([])
# model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,),name = 'LAYER1'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid',name='OUTPUT'))
model.summary()
model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])

# 이미지 생성
train_images = train_images.reshape((60000, 784))
train_images = train_images.astype('float32') / 255.0
test_images = test_images.reshape((10000, 784))
test_images = test_images.astype('float32') / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

model.fit(train_images, train_labels, epochs=5, batch_size=128)

print(model.evaluate(test_images, test_labels))
import cv2
image = cv2.imread(filename="/Users/choewonhyeong/Desktop/my_fucking_project/ABCbootcamp/AItest/Test/test.png"
                   , flags=cv2.IMREAD_GRAYSCALE)
image = cv2.resize(src = image, dsize = (28, 28))
image = image.astype('float32')
image = image.reshape(1,28*28)
image = 255- image # 반전
image /= 255.0
# 이미지 전처리 끝남
plt.imshow(image.reshape(28,28), cmap='Greys')
plt.show()

pred = model.predict(image.reshape(1, 28*28), batch_size=1)
print(f"추정된 숫자={pred.argmax()}")
