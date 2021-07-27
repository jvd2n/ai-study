from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train[0], type(x_train[0]))
# print(x_train[1], type(x_train[1]))
print(y_train[0])   # 3

print(len(x_train[0]), len(x_train[1])) # 87, 56

# print(x_train[0].shape)     # 'list' object has no attribute 'shape'

ic(x_train.shape, x_test.shape)  # (8982,) (2246,)
ic(y_train.shape, y_test.shape)  # (8982,) (2246,)

ic(type(x_train))    # <class 'numpy.ndarray'>

print("Article's Max Length: ", max(len(i) for i in x_train))   # 2376
print("Article's Avg Length: ", sum(map(len, x_train)) / len(x_train))  # 145.5

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
ic(x_train.shape, x_test.shape)
ic(type(x_train), type(x_train[0]))
ic(x_train[1])

# y 확인
ic(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

ic(y_train.shape, y_test.shape) # (8942, 46) (2246, 46)

# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

model = Sequential()
model.add(Embedding(10000, 120))
model.add(LSTM(120))
model.add(Dense(46, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es], validation_data=(x_test, y_test))

#4 Evaluate, Predict
acc = model.evaluate(x_train, y_train)[1]
ic(acc)
