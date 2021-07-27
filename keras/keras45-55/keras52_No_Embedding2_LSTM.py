from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

#1 Data
docs = ['너무 재밌어요', '참 최고예요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로예요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '배우가 잘 생기긴 했어요']

# print(len(docs))

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='post', maxlen=5)   # pre, post
print(pad_x)
print(pad_x.shape)  # (13, 5)

word_size = len(token.word_index)
print(word_size)    # 27

print(np.unique(pad_x)) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]

# One_Hot_Encoding? (13, 5) -> (13, 5, 27)
# Oxford? (13, 5, 1000000) -> 6500만개

pad_x = pad_x.reshape(13, 5, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

#2 Model
model = Sequential()
                    # 단어 사전의 개수          단어수, 길이
# model.add(Embedding(input_dim=28, output_dim=77, input_length=5))
# model.add(Embedding(28, 77, input_length=5))
# model.add(Embedding(27, 77))
model.add(Dense(32, input_shape=(5,1)))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3 Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

#4 Evaluate, Predict
acc = model.evaluate(pad_x, labels)[1]
print("acc: ", acc)
