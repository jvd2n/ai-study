from sklearn.datasets import load_boston
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (506, 13) (506,)

x = tf.compat.v1.placehloder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# results -> r2_score
