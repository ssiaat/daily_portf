import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

temp = np.linspace(0.4, 0.7, 20)
x = [temp[0], sigmoid(temp[0])]
for i in temp[1:]:
    print(i-x[0], sigmoid(i)-x[1])
    x[0] = i
    x[1] = sigmoid(i)