from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

ic(x_train.shape, y_train.shape)
ic(x_test.shape, y_test.shape)
# ic| x_train.shape: (50000, 32, 32, 3), y_train.shape: (50000, 1)
# ic| x_test.shape: (10000, 32, 32, 3), y_test.shape: (10000, 1)

ic(x_train[0])
ic(y_train[0])

plt.imshow(x_train[111])
plt.show()
