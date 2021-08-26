import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(66)

datasets = load_wine()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1, 1)

onehot_enc = preprocessing.OneHotEncoder()
y_data = onehot_enc.fit_transform(y_data).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)

print(x_train.shape, x_test.shape)  # (124, 13) (54, 13)
print(y_train.shape, y_test.shape)  # (124, 3) (54, 3)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

# w = tf.Variable(tf.random.normal([13, 3]), name='weight')
# b = tf.Variable(tf.random.normal([1, 3]), name='bias')
w = tf.Variable(tf.zeros([13, 3]), name='weight')
b = tf.Variable(tf.zeros([1, 3]), name='bias')
# zeros: 0으로 초기화 된 shape의 텐서가 만들어짐
# random.normal: 난수를 생성

# hypothesis = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) # softmax를 사용하면 총 합이 1이 됨

# catogorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-6).minimize(loss)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)

# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epochs in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if epochs % 200 == 0:
            print(epochs, cost_val)
    predicted = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(predicted, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print('accuracy_score: ', accuracy_score(y_test, y_pred))