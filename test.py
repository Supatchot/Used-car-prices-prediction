import numpy as np
import matplotlib.pyplot as plt

a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])
one = np.ones_like(a)

a = a.reshape(-1,1)
b = b.reshape(-1,1)
one = one.reshape(-1,1)

mat_y = np.concatenate((one, a), axis = 1)
mat_y = np.concatenate((mat_y, b), axis=1)
print(mat_y)

mat_x = np.concatenate((one, a, b), axis = 1)
print(mat_x)