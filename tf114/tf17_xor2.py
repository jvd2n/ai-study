# 인공지능의 겨울...
# perceptron -> mlp (multi layer perceptron)

import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# Hidden Layer 1
w1 = tf.Variable(tf.random.normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random.normal([10]), name='bias1')

# hyppthesis = x * w + b
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# Hidden Layer 2
w2 = tf.Variable(tf.random.normal([10, 4]), name='weight2')
b2 = tf.Variable(tf.random.normal([4]), name='bias2')

# hyppthesis = x * w + b
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)
# hypothesis = tf.matmul(x, w) + b    # linear

# Output Layer
w3 = tf.Variable(tf.random.normal([4, 1]), name='weight3')
b3 = tf.Variable(tf.random.normal([1]), name='bias3')

# hyppthesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)


# cost = tf.reduce_mean(tf.square(hypothesis-y)) # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 3. 훈련
for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, 'cost: ', cost_val, "\n", hy_val)

# 4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
# print(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print('*'*30)
print('예측값:\n', hy_val, '\n예측결과값: \n', c, '\nAccuracy: ', a)
sess.close()

# Accuracy:  1.0