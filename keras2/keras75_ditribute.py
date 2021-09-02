from tensorflow._api.v2 import distribute
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import cross_device_ops

#1 Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshpae(10000, 28, 28, 1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MirroredStrategy(
#     # cross_device_ops= tf.distribute.HierarchicalCopyAllReduce()
#     cross_device_ops=tf.distribute.ReductionToOneDevice()
# )
# strategy = tf.distribute.MirroredStrategy(
#     # devices=['/gpu:0']
#     # devices=['/gpu:1']
#     # devices=['/cpu', '/gpu:0']    # 느려짐
#     # devices=['/cpu', '/gpu:0', '/gpu:1']
#     devices=['/gpu:0', '/gpu:1']
# )
# strategy = tf.distribute.experimental.CentralStorageStrategy()
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    # tf.distribute.experimental.CollectiveCommunication.RING
    # tf.distribute.experimental.CollectiveCommunication.NCCL
    tf.distribute.experimental.CollectiveCommunication.AUTO
)

with strategy.scope():
    pass

