import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm
import datetime

from konlpy.tag import Okt

import sklearn
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss, accuracy_score,f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import *


DATA_IN_PATH = './dacon/_data/2_climate/'
DATA_OUT_PATH = './dacon/_output/2_climate/'

train = pd.read_csv(DATA_IN_PATH + 'train.csv')
test = pd.read_csv(DATA_IN_PATH + 'test.csv')
sample_submission = pd.read_csv(DATA_IN_PATH + 'sample_submission.csv')

length=train['과제명'].astype(str).apply(len)
plt.hist(length, bins=50, alpha=0.5, color='r', label='word')
plt.title('histogram of length of task_name')
plt.figure(figsize=(12, 5))
plt.boxplot(length, labels=['counts'], showmeans=True)
print('과제명 길이 최댓값: {}'.format(np.max(length)))
print('과제명 길이 최솟값: {}'.format(np.min(length)))
print('과제명 길이 평균값: {}'.format(np.mean(length)))
print('과제명 길이 중간값: {}'.format(np.median(length)))

length=train['요약문_연구목표'].astype(str).apply(len)
plt.hist(length, bins=50, alpha=0.5, color='r', label='word')
plt.title('histogram of length of summary_object')
plt.figure(figsize=(12, 5))
plt.boxplot(length, labels=['counts'], showmeans=True)
print('요약문_연구목표 길이 최댓값: {}'.format(np.max(length)))
print('요약문_연구목표 길이 최솟값: {}'.format(np.min(length)))
print('요약문_연구목표 길이 평균값: {}'.format(np.mean(length)))
print('요약문_연구목표 길이 중간값: {}'.format(np.median(length)))

length=train['요약문_연구내용'].astype(str).apply(len)
plt.hist(length, bins=50, alpha=0.5, color='r', label='word')
plt.title('histogram of length of summary_content')
plt.figure(figsize=(12, 5))
plt.boxplot(length, labels=['counts'], showmeans=True)
print('요약문_연구내용 길이 최댓값: {}'.format(np.max(length)))
print('요약문_연구내용 길이 최솟값: {}'.format(np.min(length)))
print('요약문_연구내용 길이 평균값: {}'.format(np.mean(length)))
print('요약문_연구내용 길이 중간값: {}'.format(np.median(length)))

length=train['요약문_기대효과'].astype(str).apply(len)
plt.hist(length, bins=50, alpha=0.5, color='r', label='word')
plt.title('histogram of length of summary_effect')
plt.figure(figsize=(12, 5))
plt.boxplot(length, labels=['counts'], showmeans=True)
print('요약문_기대효과 길이 최댓값: {}'.format(np.max(length)))
print('요약문_기대효과 길이 최솟값: {}'.format(np.min(length)))
print('요약문_기대효과 길이 평균값: {}'.format(np.mean(length)))
print('요약문_기대효과 길이 중간값: {}'.format(np.median(length)))

train=train[['과제명','label']]
test=test[['과제명']]

def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
    text=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]","", text)
    word_text=okt.morphs(text, stem=True)
    if remove_stopwords:
        word_text=[token for token in word_text if not token in stop_words]
    return word_text

stop_words=['은','는','이','가', '하','아','것','들','의','있','되','수','보','주','등','한']
okt=Okt()
clean_train_text=[]
clean_test_text=[]

for text in tqdm.tqdm(train['과제명']):
    try:
        clean_train_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text.append([])
        
for text in tqdm.tqdm(test['과제명']):
    if type(text) == str:
        clean_test_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text.append([])
        
print(len(clean_train_text))
print(len(clean_test_text))

tokenizer=Tokenizer()
tokenizer.fit_on_texts(clean_train_text)

train_sequences=tokenizer.texts_to_sequences(clean_train_text)
test_sequences=tokenizer.texts_to_sequences(clean_test_text)
word_vocab=tokenizer.word_index

#패딩 처리
train_inputs=pad_sequences(train_sequences, maxlen=40, padding='post')
test_inputs=pad_sequences(test_sequences, maxlen=40, padding='post')

print(train_inputs.shape)
print(test_inputs.shape)

labels=np.array(train['label'])
len(set(labels))

TRAIN_INPUT_DATA = 'train_input.npy'
TEST_INPUT_DATA = 'test_input.npy'

import os
if not os.path.exists(DATA_IN_PATH):
    os.makedirs(DATA_IN_PATH)
    
np.save(open(DATA_IN_PATH+TRAIN_INPUT_DATA, 'wb'), train_inputs)
np.save(open(DATA_IN_PATH+TEST_INPUT_DATA, 'wb'), test_inputs)

data_configs={}
data_configs['vocab']=word_vocab
data_configs['vocab_size'] = len(word_vocab)+1
json.dump(data_configs, open(DATA_IN_PATH+'data_configs.json', 'w'), ensure_ascii=False)

vocab_size =data_configs['vocab_size']
embedding_dim = 32
max_length = 40
oov_tok = "<OOV>"

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(46, activation='softmax')
# ])
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

# model summary
print(model.summary())

num_epochs = 30
history = model.fit(
    train_inputs, labels, 
    # epochs=num_epochs, 
    # batch_size=8,
    # verbose=2, 
    # validation_split=0.2,
)

pred=model.predict(test_inputs)
pred=tf.argmax(pred, axis=1)

sample_submission['label'] = pred

date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
sample_submission.to_csv(DATA_OUT_PATH + 'dc_cli_2_' + date_time + '.csv', index=False)
print(date_time)
