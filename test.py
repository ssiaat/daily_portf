import numpy as np
import tensorflow as tf

a = [1.0, 2.0, -3.0, 4.0,-5.0,-6.0]
a = tf.clip_by_value(a, -10, 9)
mask = tf.where((a > 0) == True, 0., -1e9)
print(tf.nn.softmax(tf.math.add(a, mask)))
