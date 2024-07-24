# 데이터 준비하기
import tensorflow as tf
data = tf.constant([[0,0],[0,1],[1,0],[1,1]],dtype = tf.int32,name = "[x1,x2]")  # constant는 2차원배열로 선언
# other_data = tf.Variable([[0,0],[0,1],[1,0],[1,1]],dtype = tf.int32,name = "[x1,x2]") # 이런형식도 가능
label = tf.constant(
    [[0],[1],[1],[0]],dtype = tf.int32, name = "y"
)
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense
model = Sequential([],name = "MODEL")
model.add(Input(shape=(2,),dtype= tf.int32, name = "INPUT"))

model.add(Dense(32,activation="relu", name = "LAYER1"))
model.add(Dense(16,activation="relu", name = "LAYER2"))
model.add(Dense(8,activation="relu", name = "LAYER3"))
model.add(Dense(1,activation="sigmoid", name = "OUTPUT"))
model.summary()
model.compile(loss='mse',optimizer = 'RMSProp')
model.fit(data,label,epochs=400)
print(model.predict(data))
