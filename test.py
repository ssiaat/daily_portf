import numpy as np
import pandas as pd
import tensorflow as tf

a = np.array([0.1,0.2,0.3,0.4])
print(tf.nn.softmax(a))
