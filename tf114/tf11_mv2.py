import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]                         # (5, 3)
y_data = [[152], [185], [180], [205], [142]]    # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([3, 1]), name='weight')    # x의 열의 개수가 w의 행의 개수와 동일해야 연산이 가능하다.
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b    # matmul: matrix의 곱셈?

print('complete')
cost = tf.reduce_mean(tf.square(hypothesis-y))  # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001) # = 1e-5
optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-5)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, "cost: ", cost_val, "\n", hy_val)

sess.close()