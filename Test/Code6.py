from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Dachshund 의 길이와 높이로 만든 데이터
dachshund_length = [77, 78, 85, 83, 73, 77, 73, 80,]
dachshund_height = [25, 28, 29, 30, 21, 22, 17, 35,]
# Samoyed 의 길이와 높이로 만든 데이터
samoyed_length = [75, 77, 86, 86, 79, 83, 83, 88,]
samoyed_height = [56, 57, 50, 53, 60, 53, 49, 61,]

# 데이터 관계도를 화면에 출력
# plt.scatter(x=dachshund_length, y=dachshund_height, c='c', marker='.')
# plt.scatter(x=samoyed_length, y=samoyed_height, c='b', marker='*')
# plt.xlabel('Heights')
# plt.ylabel('Lengths')
# plt.legend(['Dachshund', 'Samoyed'], loc='upper right')
# plt.show()

# 새로운 데이터 : 길이 79, 높이 35  : [[79, 35]]
unknown_new_dog_length = (79,)
unknown_new_dog_height = (35,)
# 데이터 관계도를 화면에 출력
plt.scatter(x=dachshund_length, y=dachshund_height, c='c', marker='.')
plt.scatter(x=samoyed_length, y=samoyed_height, c='b', marker='*')
plt.scatter(x=unknown_new_dog_length, y=unknown_new_dog_height, c='y', marker='p') # p : polygon 5각형
plt.xlabel('Heights')
plt.ylabel('Lengths')
plt.legend(['Dachshund', 'Samoyed', "Unknown New Dog"], loc='lower right')
plt.show()

# KNN Algorithm 사용하기
# [length, height] : 형태로 만들기
dachshund_data:np.ndarray = np.column_stack((dachshund_length, dachshund_height))
dachshund_data_labels:np.ndarray = np.zeros(len(dachshund_data))
print(dachshund_data_labels) # Dachshund label : 0

samoyed_data:np.ndarray = np.column_stack((samoyed_length, samoyed_height))
samoyed_data_labels:np.ndarray = np.ones(len(samoyed_data))
print(samoyed_data_labels) # Samoyed label : 1

unknown_new_dog = [[79, 35]]

# Dachshund 와 Samoyed 데이터 합체하고 target 합체하기
dogs = np.concatenate((dachshund_data, samoyed_data), axis=0)
labels = np.concatenate((dachshund_data_labels, samoyed_data_labels), axis=0)
print(dogs)
print(labels)
print(dogs.shape)
print(labels.shape)

# 레이블 (0, 1) 로 선택되어 있는 값을 문자열로 쉽게 알 수 있도록 Dictionary type 으로 표현하기
dog_classes = {0:"Dachshund", 1:"Samoyed"}
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X=dogs, y=labels)
print(knn.classes_)

y_predict = knn.predict(X=unknown_new_dog)
print(y_predict)

print(y_predict[0])

print(dog_classes[y_predict[0]])

# 닥스훈트 높이 길이 평균 데이터 만들기
dachshund_length_mean = np.mean(dachshund_length)
dachshund_height_mean = np.mean(dachshund_height)
# 사모에이드 높이 길이 평균 데이터 만들기
samoyed_length_mean = np.mean(samoyed_length)
samoyed_height_mean = np.mean(samoyed_height)

print(f"닥스훈트 평균 길이: {dachshund_length_mean}")
print(f"닥스훈트 평균 높이: {dachshund_height_mean}")

print(f"사모에이드 평균 길이:{samoyed_length_mean}")
print(f"사모에이드 평균 높이:{samoyed_height_mean}")
new_normal_dachshund_length_data = np.random.normal(dachshund_length_mean, 5.5,200)
new_normal_dachshund_height_data = np.random.normal(dachshund_height_mean, 5.5,200)
new_normal_samoyed_length_data = np.random.normal(samoyed_length_mean, 5.5,200)
new_normal_samoyed_height_data = np.random.normal(samoyed_height_mean, 5.5,200)

print(new_normal_dachshund_length_data.shape)
print(new_normal_dachshund_height_data.shape)
print(new_normal_samoyed_length_data.shape)
print(new_normal_samoyed_height_data.shape)

plt.scatter(x= new_normal_dachshund_length_data,y = new_normal_dachshund_height_data,c='b', marker='.')
plt.scatter(x= new_normal_samoyed_length_data,y = new_normal_samoyed_height_data,c='m', marker='.')
plt.show()
# 새로운 데이터 합성하고, 새로운 레이블 만들기
new_dachshund_data = np.column_stack((new_normal_dachshund_height_data, new_normal_dachshund_length_data))
new_samoyed_data = np.column_stack((new_normal_samoyed_height_data, new_normal_samoyed_length_data))
new_dachshund_labels = np.zeros(len(new_dachshund_data))# 200[0.....]
new_samoyed_labels = np.ones(len(new_samoyed_data))# 200[1 .....]

new_dogs = np.concatenate((new_dachshund_data, new_samoyed_data), axis=0)
new_labels = np.concatenate((new_dachshund_labels, new_samoyed_labels), axis=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_dogs, new_labels, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

knn = 5
knn = KNeighborsClassifier(n_neighbors=knn)
knn.fit(X_train,y_train)
print(f"훈련의 정확도: {knn.score(X_test,y_test)}")

# 예측
y_predict = knn.predict(X_test)
print(y_predict) # 예측값
print(y_test) #정답(target,label)

from sklearn.metrics import accuracy_score
print(f"테스트 정확도: {accuracy_score(y_true=y_test,y_pred=y_predict)}")
