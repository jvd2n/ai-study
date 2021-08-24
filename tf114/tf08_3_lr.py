# tf08_2 파일의 lr을 수정해서 epoch가 2000번이 아니라 100번 이하로 줄일 것
# 결과치는 step=100 이하, w=1.9999, b=0.9999

# 예측

# y = wx + b
from logging import WARNING
import tensorflow as tf

tf.set_random_seed(55)

# x_train = [1, 2, 3] # w=1, b=0
# y_train = [1, 2, 3]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable(1, dtype=tf.float32)  # 랜덤하게 넣어준 초기값
# b = tf.Variable(1, dtype=tf.float32)

W = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)

hypothesis = x_train * W + b    # 모델 구현
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

# optimizer = tf.train.AdamOptimizer(learning_rate=0.765)   # [1.9996469] [0.9877495]
optimizer = tf.train.AdamOptimizer(learning_rate=0.6221792) # [1.9999999] [0.9811333]
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.175993, use_locking=False)    # [2.0284045] [1.0071845]
# optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)    # [1.771294] [1.5077066]
# optimizer = tf.train.RMSPropOptimizer(learning_rate=0.42)   # [2.2173688] [1.2200583]
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1, 2, 3], y_train:[3, 5, 7]})
    if step % 10 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val, W_val, b_val) #, W_val, b_val)

# predict 하기 / x_test placeholder 생성

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

hypothesis_2 = x_test * W_val + b_val
x_val_1, pred_1 = sess.run([x_test, hypothesis_2], feed_dict={x_test:[4]})
x_val_2, pred_2 = sess.run([x_test, hypothesis_2], feed_dict={x_test:[5, 6]})
x_val_3, pred_3 = sess.run([x_test, hypothesis_2], feed_dict={x_test:[6, 7, 8]})
print(x_val_1, pred_1)
print(x_val_2, pred_2)
print(x_val_3, pred_3)