# np.save('./_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])

import numpy as np 
x_train = np.load('./_save/_npy/k59_3_train_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_y.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_test = np.load('./_save/_npy/k59_3_test_y.npy')

print(x_train.shape)    # (160, 150, 150, 3)
print(y_train.shape)    # (160,)
print(x_test.shape)     # (120, 150, 150, 3)
print(y_test.shape)     # (120,)