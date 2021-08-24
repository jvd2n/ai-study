# y = wx + b
from logging import WARNING
import tensorflow as tf

tf.set_random_seed(77)

x_train = [1, 2, 3] # w=1, b=0
y_train = [1, 2, 3]

# W = tf.Variable(1, dtype=tf.float32)  # 랜덤하게 넣어준 초기값
# b = tf.Variable(1, dtype=tf.float32)

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = x_train * W + b    # 모델 구현
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))