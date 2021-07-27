import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from icecream import ic

# Data
IN_PATH = './dacon/_data/'

train = pd.read_csv(IN_PATH + 'train_data.csv')
test = pd.read_csv(IN_PATH + 'test_data.csv')
submission = pd.read_csv(IN_PATH + 'sample_submission.csv')
topic_dict = pd.read_csv(IN_PATH + 'topic_dict.csv')

ic(train, test)

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean

train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))

ic(train, test)

train_text = train["cleaned_title"].tolist()
test_text = test["cleaned_title"].tolist()

# ic(train_text)

train_label = np.asarray(train.topic_idx)

tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=150000, binary=False)

tfidf.fit(train_text)

train_tf_text = tfidf.transform(train_text).astype('float32')
test_tf_text  = tfidf.transform(test_text).astype('float32')
y_train = np.array([x for x in train['topic_idx']])
ic(train_tf_text.shape, test_tf_text.shape)
# ic(train_tf_text[:1])
ic(train_label.shape)


# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(128, input_dim=150000, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(train_tf_text[:40000], train_label[:40000], epochs=10, batch_size=32, validation_data=(train_tf_text[40000:], train_label[40000:]))


# Predict
tmp_predict = model.predict(test_tf_text)
pred = np.argmax(tmp_predict, axis=1)

# Results make to_csv submissions
# ic(len(test_tf_text))
# topic = []
# for i in range(len(test_tf_text)):
#     topic.append(np.argmax(test_tf_text[i]))   # np.argmax -> 최대값의 색인 위치

submission['topic_idx'] = pred
ic(submission.shape)

import datetime
OUT_PATH = './dacon/_output/'
date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
submission.to_csv(OUT_PATH + 'CLSFY_ART_SUB_2_' + date_time + '.csv', index=False)