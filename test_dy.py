import numpy as np

random_action = np.where(np.random.random((10,)) > 0.5, 0, 1)
print(random_action)
