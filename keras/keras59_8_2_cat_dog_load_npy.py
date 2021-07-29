# np.save('./_save/_npy/k59_9_train_x.npy', arr=train_gen[0][0])
# np.save('./_save/_npy/k59_9_train_y.npy', arr=train_gen[0][1])
# np.save('./_save/_npy/k59_9_valid_x.npy', arr=valid_gen[0][0])
# np.save('./_save/_npy/k59_9_valid_y.npy', arr=valid_gen[0][1])
# np.save('./_save/_npy/k59_9_test_x.npy', arr=test_gen[0][0])
# np.save('./_save/_npy/k59_9_test_y.npy', arr=test_gen[0][1])

import numpy as np
from icecream import ic
x_train = np.load('./_save/_npy/k59_9_train_x.npy')
y_train = np.load('./_save/_npy/k59_9_train_y.npy')
x_valid = np.load('./_save/_npy/k59_9_valid_x.npy')
y_valid = np.load('./_save/_npy/k59_9_valid_y.npy')
x_test = np.load('./_save/_npy/k59_9_test_x.npy')
y_test = np.load('./_save/_npy/k59_9_test_y.npy')

ic(x_train.shape)    # (6404, 150, 150, 3)
ic(y_train.shape)    # (6404,)
ic(x_valid.shape)    # (1601, 150, 150, 3)
ic(y_valid.shape)    # (1601,)
ic(x_test.shape)    # (2023, 150, 150, 3)
ic(y_test.shape)    # (2023,)

ic(x_train)
ic(y_train[-50:])
# ic(x_train[1])
# ic(x_test[1])
# ic(y_train[1])
# ic(y_test[1])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

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
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))

# model.summary()

model.compile(
    # loss='binary_crossentropy', 
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['acc'],
)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time

start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=50,
    steps_per_epoch=32,
    validation_steps=4,
    validation_data=(x_valid, y_valid),
    callbacks=[es]
)
duration_time = time.time() - start_time

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

ic(duration_time)
ic(acc[-1])
ic(val_acc[-1])
ic(loss[-1])
ic(val_loss[-1])

y_pred = model.predict(x_test)
# ic(np.argmax(y_pred, axis=1))
# ic(np.argmax(y_pred, axis=1).shape)
# ic(np.argmax(y_test, axis=1).shape)
results = np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)
print(f'Correct_Answer_Score: {(np.count_nonzero(results == True) / results.size) * 100}')

'''
<binary>
ic| acc[-1]: 0.7891942262649536
ic| val_acc[-1]: 0.6339787840843201
ic| loss[-1]: 0.4327824115753174
ic| val_loss[-1]: 0.66398686170578
ic| y_pred: array([[0.9052797 ],
                   [0.18424067],
                   [0.5436019 ],
                   ...,
                   [0.1759071 ],
                   [0.03719163],
                   [0.21931446]], dtype=float32)

<categorical>
ic| duration_time: 631.724368095398
ic| acc[-1]: 0.8229231834411621
ic| val_acc[-1]: 0.6096189618110657
ic| loss[-1]: 0.3816540837287903
ic| val_loss[-1]: 0.7627581357955933
ic| y_pred: array([[0.5888711 , 0.56030595],
                   [0.5083643 , 0.57756186],
                   [0.61319077, 0.4564997 ],
                   ...,
                   [0.7526672 , 0.322563  ],
                   [0.45811263, 0.6674243 ],
                   [0.6220407 , 0.42301244]], dtype=float32)
                   
ic| duration_time: 801.0134689807892
ic| acc[-1]: 0.8198001384735107
ic| val_acc[-1]: 0.6008744239807129
ic| loss[-1]: 0.3784395158290863
ic| val_loss[-1]: 0.7915768027305603
Correct_Answer_Score: 61.19624320316361
'''