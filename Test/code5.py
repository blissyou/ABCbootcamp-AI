from matplotlib import pyplot as plt
import numpy as np

dach_length = [77, 78, 85, 85, 73, 77, 73, 88]  #list
dach_height = [22, 28, 29, 30, 21, 22, 17, 35]

samo_length = [75, 77, 86, 86, 79, 83, 83, 88]
samo_height = [56, 57, 50, 53, 60, 53, 49, 61]

# plt.scatter(dach_length,dach_height,color='red',marker='o',label='Dach')
# plt.scatter(samo_length,samo_height,color='blue',marker='*',label='Samo')
# plt.xlabel('length')
# plt.ylabel('height')
# plt.title('Dach and Samo')
# plt.legend(loc='upper left')
# plt.show()

d_data = np.column_stack((dach_length, dach_height))
d_label = np.zeros(len(d_data))  # 닥스훈트 길이와 높이는 0
print(d_label)
print(d_data)
s_data = np.column_stack((samo_length, samo_height))
s_label = np.ones(len(s_data))  #사모에이드의 길이와 높이는 1
print(s_label)

new_data = [[78,35]] #새로운 데이터 (어떤 종류인가요)
dog_classes = {0:"Dachhund",1:"Samoyed"}
k =3
from sklearn.neighbors import KNeighborsClassifier
knn:KNeighborsClassifier = KNeighborsClassifier(n_neighbors=3) # 쿨래스 knn는 객체

dogs = np.concatenate((d_data,s_data)) # 16마리 순서로 정해진 개 데이터
labels = np.concatenate((d_label, s_label))
knn.fit(X=dogs, y=labels)
y_predict = knn.predict(new_data)
print(f"새로운 데이터:{new_data}-판정결과 {dog_classes[y_predict[0]]}")
