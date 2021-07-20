import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
# from agent import *
# agent = Agent()

model = tf.keras.models.Sequential()
model.add(Dense(1000))


a = tf.random.normal((1, 100))
print(np.max(tf.reshape(model(a), (500, 2)), axis=1).shape)