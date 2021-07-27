import numpy as np

x_train = np.load('./_save/_npy/k55_7_x_train_data_mnist.npy')
y_train = np.load('./_save/_npy/k55_7_y_train_data_mnist.npy')
x_test = np.load('./_save/_npy/k55_7_x_test_data_mnist.npy')
y_test = np.load('./_save/_npy/k55_7_y_test_data_mnist.npy')

# print(type(x_data), type(y_data))   # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train, y_train)
print(x_train.shape, y_train.shape)

print(x_test, y_test)
print(x_test.shape, y_test.shape)