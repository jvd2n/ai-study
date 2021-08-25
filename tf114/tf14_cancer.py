from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# # 4. 평가, 예측
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
# results -> accuracy_score
