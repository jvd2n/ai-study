from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from icecream import ic

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# print(x_train[0], type(x_train[0]))
# print(x_test[0])
ic(x_train.shape, x_test.shape) # (25000,) (25000,)
ic(y_train.shape, y_test.shape) # (25000,) (25000,)
ic(type(x_train))
# ic(x_train)


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=500, padding='pre')
x_test = pad_sequences(x_test, maxlen=500, padding='pre')
ic(x_train.shape, x_test.shape)
ic(type(x_train), type(x_train[0]))
# ic(x_train[1])

word_to_index = imdb.get_word_index()
index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value+3] = key

print('빈도수 상위 1등 단어 : {}'.format(index_to_word[4]))
print('빈도수 상위 3938등 단어 : {}'.format(index_to_word[3941]))

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index]=token

print(' '.join([index_to_word[index] for index in x_train[0]]))

# ic(np.unique(y_train))
# ic(y_train[2])
# # y_train = to_categorical(y_train)
# # y_test = to_categorical(y_test)
# ic(y_train.shape, y_test.shape)
# ic(y_train[2])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Embedding

model = Sequential()
model.add(Embedding(10000, 100))
# model.add(LSTM(128))
model.add(GRU(128))
model.add(Dense(1, activation='sigmoid'))   # 이진분류 역시 다중분류이다.

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
cp = ModelCheckpoint(filepath='./_save/keras54_imdb.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=128, callbacks=[es, cp], validation_split=0.2)

acc = model.evaluate(x_train, y_train)[1]
ic(acc)

'''
ic| acc: 0.9581599831581116
ic| acc: 0.9743599891662598
ic| acc: 0.9704399704933167
'''

