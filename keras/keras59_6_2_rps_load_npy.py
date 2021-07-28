# np.save('./_save/_npy/k59_7_train_x.npy', arr=trainGen[0][0])
# np.save('./_save/_npy/k59_7_train_y.npy', arr=trainGen[0][1])
# np.save('./_save/_npy/k59_7_test_x.npy', arr=testGen[0][0])
# np.save('./_save/_npy/k59_7_test_y.npy', arr=testGen[0][1])

import numpy as np
from icecream import ic
x_train = np.load('./_save/_npy/k59_7_train_x.npy')
y_train = np.load('./_save/_npy/k59_7_train_y.npy')
x_test = np.load('./_save/_npy/k59_7_test_x.npy')
y_test = np.load('./_save/_npy/k59_7_test_y.npy')

print(x_train.shape)    # (2648, 150, 150, 3)
print(y_train.shape)    # (2648, 2)
print(x_test.shape)     # (661, 150, 150, 3)
print(y_test.shape)     # (661, 2)

ic(y_train[0])


# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(InputLayer(input_shape=(150,150,3)))
model.add(Conv2D(16, (3, 3), (1, 1), 'same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(32, (3, 3), (1, 1), 'same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), (1, 1), 'same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

# model.summary()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['acc'],
)

hist = model.fit(
    x_train, y_train,
    epochs=30,
    steps_per_epoch=32,
    validation_steps=4,
    validation_data=(x_test, y_test),
    callbacks=[es]
)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_acc']

ic(loss[-1])
ic(val_acc[-1])
ic(acc[-1])
# ic(val_acc[:-1])

'''
ic| loss[-1]: 0.1396760791540146
ic| val_acc[-1]: 0.4503968358039856
ic| acc[-1]: 0.9479166865348816
'''
