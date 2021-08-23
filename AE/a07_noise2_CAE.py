import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train_DNN = x_train.reshape(60000, 28, 28, 1).astype('float')/255
x_train_CNN = x_train.reshape(60000,  28 * 28).astype('float')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float')/255

x_train_noised = x_train_DNN + np.random.normal(0, 0.1, size=x_train_DNN.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D, Flatten

def autoencoder1(hidden_layer_size): # 기본적인 오토인코더
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2), input_shape=(28, 28, 1),
                    activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Conv2D(256, (2,2), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, (2,2), activation='relu', padding='same'))   
    model.add(Flatten())
    model.add(Dense(784, activation='sigmoid'))
    return model


def autoencoder2(hidden_layer_size): # 딥하게 구성
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2), input_shape=(28, 28, 1),
                    activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(MaxPool2D())
    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(784, activation='sigmoid'))
    return model

model1 = autoencoder1(hidden_layer_size=128)   # pca 95% -> 154
model2 = autoencoder2(hidden_layer_size=512)

model1.compile(optimizer='adam', loss='mse')
model2.compile(optimizer='adam', loss='mse')

model1.fit(x_train_DNN, x_train_CNN, epochs=10, batch_size=32)
model2.fit(x_train_DNN, x_train_CNN, epochs=10, batch_size=32)

output1 = model1.predict(x_test)
output2 = model2.predict(x_test)


from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))

random_images = random.sample(range(output1.shape[0]), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output1[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT_1", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output2[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT_2", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()