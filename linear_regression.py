import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense

x = [[1],[2],[3],[4],[5],[6],[7]]
y = [2,4,6,8,10,12,14]

model = tf.keras.Sequential([Dense(1,activation='linear')])
model.compile(optimizer=tf.optimizers.SGD(0.018),loss='mse')
history=model.fit(x,y,epochs=500)
loss = history.history['loss']
prediction = [10]
print(model.predict(prediction))

plt.subplot(2,2,1)
plt.plot(x,y,color='green')
plt.subplot(2,2,2)
plt.plot(loss,color='red')
plt.show()