import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
# from agent import *
# agent = Agent()

a = np.array([10,3, 0])
print(np.where(a==0, 1, a))
b = np.array([5,4])
print(a.clip(0,b))