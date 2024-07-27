import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# 얼굴 이미지를 저장할 리스트
face_images = []
for i in range(15):
    file = './faces/' + 'img{0:02d}.jpg'.format(i + 1)
    img = cv2.imread(file)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_images.append(img)
    else:
        print(f"Error: Unable to read image at {file}")

# 동물 이미지를 저장할 리스트
animal_images = []
for i in range(15):
    file = './animals/' + 'img{0:02d}.jpg'.format(i + 1)
    img = cv2.imread(file)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        animal_images.append(img)
    else:
        print(f"Error: Unable to read image at {file}")

def plot_images(n_row: int, n_col: int, images: list[np.ndarray]) -> None:
    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i in range(n_row * n_col):
        if i < len(images):
            img = images[i]
            row = i // n_col
            col = i % n_col
            ax[row, col].imshow(img)
            ax[row, col].axis('off')
        else:
            ax[row, col].axis('off')

# 얼굴 이미지 플롯
plot_images(3, 5, face_images)
plt.show()

# 동물 이미지 플롯
plot_images(3, 5, animal_images)
plt.show()

X = face_images + animal_images
y = [[1,0]]*len(face_images) + [[0,1]]*len(animal_images)
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

# CNN 모델 만들기
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax')
], name="FACE_DETECTOR")

model.summary()
#
# # 모델 컴파일 및 학습
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(x=X, y=y, epochs=10)
# model.save('face_detector.keras')

# 예제 파일을 이용해서 이미지 테스트
examples_images = []
for i in range(10):
    file = './examples/' + 'img{0:02d}.jpg'.format(i + 1)
    img = cv2.imread(file)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        examples_images.append(img)
    else:
        print(f"Error: Unable to read image at {file}")
examples_images = np.array(examples_images)
plot_images(5, 2, examples_images)
plt.show()

# 모델 불러오기 및 예측
model2 = tf.keras.models.load_model("face_detector.keras")
predict_images = model2.predict(examples_images)

fig, ax = plt.subplots(2, 5, figsize=(10, 4))
for i in range(2):
    for j in range(5):
        axis = ax[i, j]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.imshow(examples_images[i * 5 + j])
        title = "Face" if predict_images[i * 5 + j][0] > 0.5 else "i don't know"
        axis.set_title(title)
plt.show()

# 예측 결과 플롯
plt.plot(predict_images)
plt.show()
