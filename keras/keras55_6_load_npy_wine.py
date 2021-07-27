import numpy as np

x_data = np.load('./_save/_npy/k55_6_x_data_wine.npy')
y_data = np.load('./_save/_npy/k55_6_y_data_wine.npy')

# print(type(x_data), type(y_data))   # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)

