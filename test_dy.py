import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

a = np.array([[[[1,2,3,4]],[[5,6,7,8]]], [[[9,10,11,12]],[[13,14,15,16]]], [[[17,18,19,20]],[[21,22,23,24]]]])

a = a.swapaxes(0,1)
a = np.split(a, 2, axis=0)

print(a[0].shape)
# print([a[:].shape])