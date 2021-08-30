import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
tf.compat.v1.set_random_seed(66)

#1 Data
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2 Modelng
W1 = tf.compat.v1.get_variable(name='w1', shape=[3, 3, 1, 32])
                            # kernel_size, input, output
L1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1,
#                  padding='valid', input_shape=(28, 28, 1)))

print(W1)   # (3, 3, 1, 32)
print(L1)   # (?, 28, 28, 32)

############ get_variable 연구 ############
# w2 = tf.Variable(tf.random.normal([3, 3, 1, 32]), dtype=tf.float32) # [3, 3, 1, 32]?
# w3 = tf.Variable([1], dtype=tf.float32)
# # get_variable & Variable: get_variable은 자동으로 초기값을 넣어준다.

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# print(np.min(sess.run(w1)))
# print(np.max(sess.run(w1)))
# print(np.mean(sess.run(w1)))
# print(np.median(sess.run(w1)))
# print(w1)
##########################################
