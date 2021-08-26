import numpy as np
import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]] # (N, 4) x w = (N, 3)   ->  w: (4, 3)
y_data = [[0, 0, 1],    # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],    # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],    # 0
          [1, 0, 0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random.normal([4, 3]), name='weight')
b = tf.Variable(tf.random.normal([1, 3]), name='bias')

# hypothesis = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) # softmax를 사용하면 총 합이 1이 됨

# catogorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epochs in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if epochs % 200 == 0:
            print(epochs, cost_val)
    
    # predict
    results = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
    print(results, sess.run(tf.argmax(results, 1)))

'''
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print('*'*30)
print('예측값:\n', hy_val, '\n예측결과값: \n', c, '\nAccuracy: ', a)
sess.close()
'''