import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(66)

datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)

onehot_enc = preprocessing.OneHotEncoder()
y_train = onehot_enc.fit_transform(y_train).toarray()
print(x_train.shape, x_test.shape)  # (105, 4) (45, 4)
print(y_train.shape, y_test.shape)  # (105, 3) (45, 3)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random.normal([4, 3]), name='weight')
b = tf.Variable(tf.random.normal([1, 3]), name='bias')

# hypothesis = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) # softmax를 사용하면 총 합이 1이 됨

# catogorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epochs in range(2001):
        _, cost_val, hy_val = sess.run([optimizer, loss, hypothesis], feed_dict={x:x_train, y:y_train})
        if epochs % 200 == 0:
            print(epochs, cost_val)
    predicted = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(predicted, axis=1)
    print('accuracy_score: ', accuracy_score(y_test, y_pred))