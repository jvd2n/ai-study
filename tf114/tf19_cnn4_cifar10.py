# 0.7 이상

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   # False
print(tf.__version__)           # 1.14.0 -> 2.4.0

tf.compat.v1.set_random_seed(66)


# 1 Data
# from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 10)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 10)

x_train = x_train.reshape(-1, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 2 Modelng
# layer1
w1 = tf.compat.v1.get_variable(name='w1', shape=[3, 3, 3, 32])
# kernel_size, input, output
print(w1)   # (3, 3, 1, 32)
layer1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
print(layer1)  # VALID -> (?, 26, 26, 32) // SAME -> (?, 28, 28, 32)
layer1 = tf.nn.relu(layer1)
l1_maxpool = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[
                            1, 2, 2, 1], padding='SAME')


# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1,
#                  padding='valid', input_shape=(28, 28, 1),
#                  activation='relu'))
# model.add(MaxPool2D())

print(layer1)   # (?, 28, 28, 32)
print(l1_maxpool)   # (?, 14, 14, 32)

# layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 32, 64])
layer2 = tf.nn.conv2d(l1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME')
layer2 = tf.nn.relu(layer2)
l2_maxpool = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[
                            1, 2, 2, 1], padding='SAME')

print(layer2)   # (?, 14, 14, 64)
print(l2_maxpool)   # (?, 7, 7, 64)

# layer3
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 128])
layer3 = tf.nn.conv2d(l2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME')
layer3 = tf.nn.relu(layer3)
l3_maxpool = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[
                            1, 2, 2, 1], padding='SAME')

print(layer3)   # (?, 7, 7, 128)
print(l3_maxpool)   # (?, 4, 4, 128)

# layer4
# initializer=tf.compat.v1.contrib.layers.xavier_initializer())
w4 = tf.compat.v1.get_variable('w4', shape=[2, 2, 128, 64],)
layer4 = tf.nn.conv2d(l3_maxpool, w4, strides=[1, 1, 1, 1], padding='VALID')
layer4 = tf.nn.relu(layer4)
l4_maxpool = tf.nn.max_pool(layer4, ksize=[1, 2, 2, 1], strides=[
                            1, 2, 2, 1], padding='SAME')

print(layer4)   # (?, 3, 3, 64)
print(l4_maxpool)   # (?, 2, 2, 64)

# Flatten
layer_flat = tf.reshape(l4_maxpool, [-1, 2*2*64])
print('Flatten(): ', layer_flat)  # (?, 256)

# layer5 DNN
# initializer=tf.compat.v1.contrib.layers.xavier_initializer())
w5 = tf.compat.v1.get_variable('w5', shape=[2*2*64, 64],)
b5 = tf.Variable(tf.random.normal([64]), name='b1')
layer5 = tf.matmul(layer_flat, w5) + b5
layer5 = tf.nn.relu(layer5)
# layer5 = tf.nn.dropout(layer5, rate=1-0.2) # keep_prob=0.2)
print(layer5)   # (?, 64)

# layer6 DNN
# initializer=tf.compat.v1.contrib.layers.xavier_initializer())
w6 = tf.compat.v1.get_variable('w6', shape=[64, 32],)
b6 = tf.Variable(tf.random.normal([32]), name='b2')
layer6 = tf.matmul(layer5, w6) + b6
layer6 = tf.nn.relu(layer6)
# layer6 = tf.nn.dropout(layer6, rate=1-0.2) # keep_prob=0.2)
print(layer6)   # (?, 32)

# layer7 Softmax
w7 = tf.compat.v1.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.random.normal([10]), name='b3')
layer7 = tf.matmul(layer6, w7) + b7
hypothesis = tf.nn.softmax(layer7)
print(hypothesis)   # (?, 10)

# 3 Compile
# categorical crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# total_batch = int(len(x_train)/batch_size)

for epoch in range(training_epochs):
    avg_loss = 0

    for i in range(total_batch):
        # 몇 번 돌까?
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x: batch_x, y: batch_y}

        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss += batch_loss/total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'loss: {:.9f}'.format(avg_loss))

print('Training Complete')

prediction = tf.equal(tf.compat.v1.argmax(
    hypothesis, 1), tf.compat.v1.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc: ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
# Acc:  0.7211
