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
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

DATA_IN_PATH = './dacon/_data/2_climate/'
DATA_OUT_PATH = './dacon/_output/2_climate/'

train = pd.read_csv(DATA_IN_PATH + 'train.csv')
test = pd.read_csv(DATA_IN_PATH + 'test.csv')
sample_submission = pd.read_csv(DATA_IN_PATH + 'sample_submission.csv')

length = train['과제명'].astype(str).apply(len)
plt.hist(length, bins=50, alpha=0.5, color='r', label='word')
plt.title('histogram of length of task_name')
plt.figure(figsize=(12, 5))
plt.boxplot(length, labels=['counts'], showmeans=True)
print('과제명 길이 최댓값: {}'.format(np.max(length)))
print('과제명 길이 최솟값: {}'.format(np.min(length)))
print('과제명 길이 평균값: {}'.format(np.mean(length)))
print('과제명 길이 중간값: {}'.format(np.median(length)))

length = train['요약문_연구목표'].astype(str).apply(len)
plt.hist(length, bins=50, alpha=0.5, color='r', label='word')
plt.title('histogram of length of summary_object')
plt.figure(figsize=(12, 5))
plt.boxplot(length, labels=['counts'], showmeans=True)
print('요약문_연구목표 길이 최댓값: {}'.format(np.max(length)))
print('요약문_연구목표 길이 최솟값: {}'.format(np.min(length)))
print('요약문_연구목표 길이 평균값: {}'.format(np.mean(length)))
print('요약문_연구목표 길이 중간값: {}'.format(np.median(length)))

length = train['요약문_연구내용'].astype(str).apply(len)
plt.hist(length, bins=50, alpha=0.5, color='r', label='word')
plt.title('histogram of length of summary_content')
plt.figure(figsize=(12, 5))
plt.boxplot(length, labels=['counts'], showmeans=True)
print('요약문_연구내용 길이 최댓값: {}'.format(np.max(length)))
print('요약문_연구내용 길이 최솟값: {}'.format(np.min(length)))
print('요약문_연구내용 길이 평균값: {}'.format(np.mean(length)))
print('요약문_연구내용 길이 중간값: {}'.format(np.median(length)))

train = train[['과제명', 'label']]
test = test[['과제명']]


def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]", "", text)
    word_text = okt.morphs(text, stem=True)
    if remove_stopwords:
        word_review = [token for token in word_text if not token in stop_words]
    return word_review


stop_words = ['은', '는', '이', '가', '하', '아', '것',
              '들', '의', '있', '되', '수', '보', '주', '등', '한']
okt = Okt()
clean_train_text = []
clean_test_text = []

for text in tqdm.tqdm(train['과제명']):
    try:
        clean_train_text.append(preprocessing(
            text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text.append([])

for text in tqdm.tqdm(test['과제명']):
    if type(text) == str:
        clean_test_text.append(preprocessing(
            text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text.append([])

print(len(clean_train_text))
print(len(clean_test_text))

vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
train_features = vectorizer.fit_transform(clean_train_text)
test_features = vectorizer.transform(clean_test_text)

TEST_SIZE = 0.2
RANDOM_SEED = 42

train_x, eval_x, train_y, eval_y = train_test_split(
    train_features, train['label'], test_size=TEST_SIZE, random_state=RANDOM_SEED)

model = RandomForestClassifier(n_estimators=100)

model.fit(train_x, train_y)

model.score(eval_x, eval_y)

# y_pred = model.predict(test_features)

sample_submission['label'] = model.predict(test_features)

date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
sample_submission.to_csv(DATA_OUT_PATH + 'dc_cli_3_' +
                         date_time + '.csv', index=False)
print(date_time)
