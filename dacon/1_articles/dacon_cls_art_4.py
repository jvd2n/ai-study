import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertModel, TFBertModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from transformers.models.bert.tokenization_bert import BertTokenizer

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

tf.random.set_seed(1234)
np.random.seed(1234)

BATCH_SIZE = 32
NUM_EPOCHS = 3
VALID_SPLIT = 0.2
MAX_LEN = 44
DATA_IN_PATH = './dacon/_data/'
DATA_OUT_PATH = './dacon/_output/'
PATH = './dacon/_data/'
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir='./dacon/bert_ckpt', do_lower_case=False)

train_data = pd.read_csv(PATH + 'train_data.csv')
test_data = pd.read_csv(PATH + 'test_data.csv')
submission = pd.read_csv(PATH + 'sample_submission.csv')

def bert_tokenizer(sent, MAX_LEN):
    encoded_dict = tokenizer.encode_plus(
        text = sent,
        add_special_tokens = True,
        max_length = MAX_LEN,
        pad_to_max_length = True,
        return_attention_mask = True,
    )
    
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']
    
    return input_id, attention_mask, token_type_id

input_ids = []
attention_masks = []
token_type_ids = []
train_data_labels = []

for train_sent, train_label in tqdm(zip(train_data["title"], train_data["topic_idx"]), total=len(train_data)):
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer(train_sent, MAX_LEN)
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        train_data_labels.append(train_label)

    except Exception as e:
        print(e)
        print(train_sent)
        pass

train_movie_input_ids = np.array(input_ids, dtype=int)
train_movie_attention_masks = np.array(attention_masks, dtype=int)
train_movie_type_ids = np.array(token_type_ids, dtype=int)
train_movie_inputs = (train_movie_input_ids, train_movie_attention_masks, train_movie_type_ids)

train_data_labels = np.asarray(train_data_labels, dtype=np.int32) #레이블 토크나이징 리스트

print("# sents: {}, # labels: {}".format(len(train_movie_input_ids), len(train_data_labels)))

class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(TFBertClassifier, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class, 
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range), 
                                                name="classifier")
        
    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        #outputs 값: # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits

cls_model = TFBertClassifier(
    model_name='bert-base-multilingual-cased',
    dir_path='bert_ckpt',
    num_class=7
)

optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model_name = "tf2_bert_classify_article"

# overfitting을 막기 위한 ealrystop 추가
es = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=4)
# min_delta: the threshold that triggers the termination (acc should at least improve 0.0001)
# patience: no improvment epochs (patience = 1, 1번 이상 상승이 없으면 종료)\

checkpoint_path = os.path.join(DATA_OUT_PATH, model_name, 'dacon_clf_atcl_weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create path if exists
if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))
    
cp = ModelCheckpoint(
    checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)

history = cls_model.fit(
    train_movie_inputs, 
    train_data_labels,
    verbose=1,
    epochs=NUM_EPOCHS, 
    batch_size=BATCH_SIZE,
    validation_split = VALID_SPLIT, 
    callbacks=[es, cp]
)

#steps_for_epoch

print(history.history)

input_ids = []
attention_masks = []
token_type_ids = []
test_data_labels = []

for test_sent in tqdm(test_data["title"]):
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer(test_sent, MAX_LEN)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
    except Exception as e:
        print(e)
        print(test_sent)
        pass

test_article_input_ids = np.array(input_ids, dtype=int)
test_article_attention_masks = np.array(attention_masks, dtype=int)
test_article_type_ids = np.array(token_type_ids, dtype=int)
test_article_inputs = (test_article_input_ids, test_article_attention_masks, test_article_type_ids)

results = cls_model.predict(test_article_inputs, batch_size=1024)
#results=tf.argmax(results, axis=1)

topic = []
for i in range(len(results)):
    topic.append(np.argmax(results[i]))

date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
submission['topic_idx']=topic
submission.to_csv(DATA_OUT_PATH + 'bert_mod_' + date_time + '.csv', index=False)