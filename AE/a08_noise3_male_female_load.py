# keras61_5 남자 여자 데이터에 노이즈를 넣고
# 기미 주근깨 여드름 제거



import numpy as np
from icecream import ic
x_train = np.load('./_save/_npy/a08_noise3_train_x.npy')
y_train = np.load('./_save/_npy/a08_noise3_train_y.npy')
x_test = np.load('./_save/_npy/a08_noise3_valid_x.npy')
y_test = np.load('./_save/_npy/a08_noise3_valid_y.npy')

ic(x_train.shape)    # (2648, 150, 150, 3)
ic(y_train.shape)    # (2648,)
ic(x_test.shape)    # (661, 150, 150, 3)
ic(y_test.shape)    # (661,)

# x_train = x_train.reshape(-1, 30, 30, 3).astype('float')/255
# x_train_2 = x_train.reshape(-1, 30*30*3).astype('float')/255
# x_test = x_test.reshape(-1, 30, 30, 3).astype('float')/255

x_train = x_train.reshape(-1, 150, 150, 3).astype('float')/255
x_test = x_test.reshape(-1, 150, 150, 3).astype('float')/255

x_train_noised = x_train + np.random.normal(0, 0.05, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.05, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

ic(x_train.shape, x_train_noised.shape)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D, Flatten, Dropout

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2,2), input_shape=(150, 150, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(1,1))
    model.add(Dropout(0.9))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(Dropout(0.9))
    model.add(MaxPool2D(1,1))
    model.add(Conv2D(3, (2, 2), activation='sigmoid', padding='same'))
    # model.add(Dense(units=hidden_layer_size, input_shape=(120*120*3,), activation='relu'))
    # model.add(Dense(120*120*3, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)  # pca 95% -> 154

model.compile(loss='binary_crossentropy', optimizer='adam') #, metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=10, verbose=1)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(3, 5, figsize=(20, 10))

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150, 3) * 255, vmin=0, vmax=255)
    # ax.imshow((x_test[random_images[i]].reshape(120, 120, 3) * 255).astype('uint8'))
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈를 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3) * 255, vmin=0, vmax=255)
    # ax.imshow((x_test_noised[random_images[i]].reshape(120, 120, 3) * 255).astype('uint8'))
    if i == 0:
        ax.set_ylabel("NOISED", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(150, 150, 3) * 255, vmin=0, vmax=255)
    # ax.imshow((output[random_images[i]].reshape(120, 120, 3) * 255).astype('uint8'))
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

'''
model = Sequential()
model.add(InputLayer(input_shape=(150, 150, 3)))
model.add(Conv2D(16, (3, 3), (1, 1), 'same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), (1, 1), 'same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

model.compile(
    loss='binary_crossentropy', 
    optimizer='adam',
    metrics=['acc'],
)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

hist = model.fit(
    x_train, y_train,
    epochs=50,
    steps_per_epoch=32,
    validation_steps=4,
    validation_data=(x_valid, y_valid),
    callbacks=[es]
)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

ic(acc[-1])
ic(val_acc[-1])
ic(loss[-1])
ic(val_loss[-1])

y_pred = model.predict(x_test)
ic(y_pred)
'''