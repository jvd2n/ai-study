from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
from icecream import ic

datasets = load_iris()

x_data = datasets.data
y_data = datasets.target

print(type(x_data), type(y_data))   
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_2_x_data_iris.npy', arr=x_data)
np.save('./_save/_npy/k55_2_y_data_iris.npy', arr=y_data)


datasets = load_boston()
x_data = datasets.data
y_data = datasets.target

np.save('./_save/_npy/k55_3_x_data_boston.npy', arr=x_data)
np.save('./_save/_npy/k55_3_y_data_boston.npy', arr=y_data)


datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target

np.save('./_save/_npy/k55_4_x_data_cancer.npy', arr=x_data)
np.save('./_save/_npy/k55_4_y_data_cancer.npy', arr=y_data)


datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target

np.save('./_save/_npy/k55_5_x_data_diabetes.npy', arr=x_data)
np.save('./_save/_npy/k55_5_y_data_diabetes.npy', arr=y_data)


datasets = load_wine()
x_data = datasets.data
y_data = datasets.target

np.save('./_save/_npy/k55_6_x_data_wine.npy', arr=x_data)
np.save('./_save/_npy/k55_6_y_data_wine.npy', arr=y_data)


# datasets = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# ic(type(x_train))
# ic| type(x_train): <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_7_x_train_data_mnist.npy', arr=x_train)
np.save('./_save/_npy/k55_7_y_train_data_mnist.npy', arr=y_train)
np.save('./_save/_npy/k55_7_x_test_data_mnist.npy', arr=x_test)
np.save('./_save/_npy/k55_7_y_test_data_mnist.npy', arr=y_test)


# datasets = fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# ic(type(x_train))
# ic| type(x_train): <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_8_x_train_data_fashion.npy', arr=x_train)
np.save('./_save/_npy/k55_8_y_train_data_fashion.npy', arr=y_train)
np.save('./_save/_npy/k55_8_x_test_data_fashion.npy', arr=x_test)
np.save('./_save/_npy/k55_8_y_test_data_fashion.npy', arr=y_test)


# datasets = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# ic(type(x_train))
# ic| type(x_train): <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_9_x_train_data_cifar10.npy', arr=x_train)
np.save('./_save/_npy/k55_9_y_train_data_cifar10.npy', arr=y_train)
np.save('./_save/_npy/k55_9_x_test_data_cifar10.npy', arr=x_test)
np.save('./_save/_npy/k55_9_y_test_data_cifar10.npy', arr=y_test)


datasets = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# ic(type(x_train))
# ic| type(x_train): <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_10_x_train_data_cifar100.npy', arr=x_train)
np.save('./_save/_npy/k55_10_y_train_data_cifar100.npy', arr=y_train)
np.save('./_save/_npy/k55_10_x_test_data_cifar100.npy', arr=x_test)
np.save('./_save/_npy/k55_10_y_test_data_cifar100.npy', arr=y_test)
