# 덧셈 뺄셈 곱셈 나눗셈 만들기

import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.divide(node1, node2)

sess = tf.Session()

print(sess.run(node3), sess.run(node4), sess.run(node5), sess.run(node6))