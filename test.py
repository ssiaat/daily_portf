import numpy as np
import tensorflow as tf

a = np.array([1,2,3, -4])
a = tf.clip_by_value(a, 0, 10)
print(a.shape[0])
# print(a[a<0.])
