from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


print('numpy verison :', np.__version__)
print('pandas version :', pd.__version__)
print('opencv version :', cv2.__version__)
print('rdkit version :', rdkit.__version__)
print('torch version :', torch.__version__)
print(torch.cuda.is_available())

train = pd.read_csv('./dacon/_data/3_scientific/train.csv')
dev = pd.read_csv('./dacon/_data/3_scientific/dev.csv')

print(train.head())
print(dev.head())

train = pd.concat([train, dev])
train['S1_energy(eV)'].hist(bins=100, alpha=0.5)
train['T1_energy(eV)'].hist(bins=100, alpha=0.5)
# plt.show()

for idx, row in tqdm(train.iterrows()):
    file = row['uid']
    smiles = row['SMILES']
    m = Chem.MolFromSmiles(smiles)
    if m != None:
        img = Draw.MolToImage(m, size=(300, 300))
        img.save(f'./dacon/_data/3_scientific/train_imgs/{file}.png')

sample_img = cv2.imread('./dacon/_data/3_scientific/train_imgs/dev_0.png')
plt.imshow(sample_img)
# plt.show()

device = torch.device("cuda:0")
BATCH_SIZE = 48
EPOCHS = 25
num_layers = 1
dropout_rate = 0.2
embedding_dim = 128
learning_rate = 1e-4
vision_pretrain = True
save_path = f'./dacon/_data/3_scientific/models/best_model.pt'


class SMILES_Tokenizer():
    def __init__(self, max_length):
        self.txt2idx = {}
        self.idx2txt = {}
        self.max_length = max_length

    def fit(self, SMILES_list):
        unique_char = set()
        for smiles in SMILES_list:
            for char in smiles:
                unique_char.add(char)
        unique_char = sorted(list(unique_char))
        for i, char in enumerate(unique_char):
            self.txt2idx[char] = i+2
            self.idx2txt[i+2] = char

    def txt2seq(self, texts):
        seqs = []
        for text in tqdm(texts):
            seq = [0]*self.max_length
            for i, t in enumerate(text):
                if i == self.max_length:
                    break
                try:
                    seq[i] = self.txt2idx[t]
                except:
                    seq[i] = 1
            seqs.append(seq)
        return np.array(seqs)


max_len = train.SMILES.str.len().max()
print(max_len)

tokenizer = SMILES_Tokenizer(max_len)
tokenizer.fit(train.SMILES)

seqs = tokenizer.txt2seq(train.SMILES)
labels = train[['S1_energy(eV)', 'T1_energy(eV)']].to_numpy()
imgs = ('./dacon/_data/3_scientific/train_imgs/'+train.uid+'.png').to_numpy()

imgs, seqs, labels = shuffle(imgs, seqs, labels, random_state=42)

train_imgs = imgs[:27000]
train_seqs = seqs[:27000]
train_labels = labels[:27000]
val_imgs = imgs[27000:]
val_seqs = seqs[27000:]
val_labels = labels[27000:]

print(train_imgs.shape, train_seqs.shape, train_labels.shape,
      val_imgs.shape, val_seqs.shape, val_labels.shape)


class CustomDataset(Dataset):
    def __init__(self, imgs, seqs, labels=None, mode='train'):
        self.mode = mode
        self.imgs = imgs
        self.seqs = seqs
        if self.mode == 'train':
            self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i]).astype(np.float32)/255
        img = np.transpose(img, (2, 0, 1))
        if self.mode == 'train':
            return {
                'img': torch.tensor(img, dtype=torch.float32),
                'seq': torch.tensor(self.seqs[i], dtype=torch.long),
                'label': torch.tensor(self.labels[i], dtype=torch.float32)
            }
        else:
            return {
                'img': torch.tensor(img, dtype=torch.float32),
                'seq': torch.tensor(self.seqs[i], dtype=torch.long),
            }


train_dataset = CustomDataset(train_imgs, train_seqs, train_labels)
val_dataset = CustomDataset(val_imgs, val_seqs, val_labels)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
# train_dataloader = torch.tensor(train_dataset.values).float()
# val_dataloader = torch.tensor(val_dataset.values).float()

sample_batch = next(iter(train_dataloader))

print(sample_batch['img'].size(),
      sample_batch['seq'].size(), sample_batch['label'].size())
print(sample_batch['img'].dtype, sample_batch['seq'].dtype,
      sample_batch['label'].dtype)


class CNN_Encoder(nn.Module):
    def __init__(self, embedding_dim, rate):
        super(CNN_Encoder, self).__init__()
        model = models.resnet50(pretrained=vision_pretrain)
        modules = list(model.children())[:-2]
        self.feature_extract_model = nn.Sequential(*modules)
        self.dropout1 = nn.Dropout(rate)
        self.fc = nn.Linear(2048, embedding_dim)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        x = self.feature_extract_model(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(3))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc(x))
        x = self.dropout2(x)
        return x


class RNN_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_layers, rate):
        super(RNN_Decoder, self).__init__()
        self.embedding = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(rate)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers)
        self.final_layer = nn.Linear((max_len+100)*embedding_dim, 2)

    def forward(self, enc_out, dec_inp):
        embedded = self.embedding(dec_inp)
        embedded = self.dropout(embedded)
        embedded = torch.cat([enc_out, embedded], dim=1)
        hidden, _ = self.lstm(embedded)
        hidden = hidden.view(hidden.size(0), -1)
        output = nn.ReLU()(self.final_layer(hidden))
        return output


class CNN2RNN(nn.Module):
    def __init__(self, embedding_dim, max_len, num_layers, rate):
        super(CNN2RNN, self).__init__()
        self.cnn = CNN_Encoder(embedding_dim, rate)
        self.rnn = RNN_Decoder(max_len, embedding_dim, num_layers, rate)

    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)

        return output


model = CNN2RNN(embedding_dim=embedding_dim, max_len=max_len,
                num_layers=num_layers, rate=dropout_rate)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()


def train_step(batch_item, epoch, batch, training):
    img = batch_item['img'].to(device)
    seq = batch_item['seq'].to(device)
    label = batch_item['label'].to(device)
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img, seq)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        return loss
    else:
        model.eval()
        with torch.no_grad():
            output = model(img, seq)
            loss = criterion(output, label)

        return loss


loss_plot, val_loss_plot = [], []

for epoch in range(EPOCHS):
    total_loss, total_val_loss = 0, 0

    tqdm_dataset = tqdm(enumerate(train_dataloader))
    training = True
    for batch, batch_item in tqdm_dataset:
        batch_loss = train_step(batch_item, epoch, batch, training)
        total_loss += batch_loss

        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Loss': '{:06f}'.format(batch_loss.item()),
            'Total Loss': '{:06f}'.format(total_loss/(batch+1))
        })
    loss_plot.append(total_loss/(batch+1))

    tqdm_dataset = tqdm(enumerate(val_dataloader))
    training = False
    for batch, batch_item in tqdm_dataset:
        batch_loss = train_step(batch_item, epoch, batch, training)
        total_val_loss += batch_loss

        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Val Loss': '{:06f}'.format(batch_loss.item()),
            'Total Val Loss': '{:06f}'.format(total_val_loss/(batch+1))
        })
    val_loss_plot.append(total_val_loss/(batch+1))

    if np.min(val_loss_plot) == val_loss_plot[-1]:
        torch.save(model, save_path)

plt.plot(loss_plot, label='train_loss')
plt.plot(val_loss_plot, label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss(mae)')
plt.legend()
# plt.show()

model = torch.load(save_path)
test = pd.read_csv('./dacon/_data/3_scientific/test.csv')
submission = pd.read_csv('./dacon/_data/3_scientific/sample_submission.csv')

for idx, row in tqdm(test.iterrows()):
    file = row['uid']
    smiles = row['SMILES']
    m = Chem.MolFromSmiles(smiles)
    if m != None:
        img = Draw.MolToImage(m, size=(300, 300))
        img.save(f'./dacon/_data/3_scientific/test_imgs/{file}.png')

test_seqs = tokenizer.txt2seq(test.SMILES)
test_imgs = ('./dacon/_data/3_scientific/test_imgs/' +
             test.uid+'.png').to_numpy()

test_dataset = CustomDataset(
    imgs=test_imgs, seqs=test_seqs, labels=None, mode='test')
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=16)


def predict(dataset):
    model.eval()
    result = []
    for batch_item in dataset:
        img = batch_item['img'].to(device)
        seq = batch_item['seq'].to(device)
        with torch.no_grad():
            output = model(img, seq)
        output = output.cpu().numpy()
        gap = output[:, 0] - output[:, 1]
        gap = np.where(gap < 0, 0, gap)
        result.extend(list(gap))
    return result


pred = predict(test_dataloader)
submission['ST1_GAP(eV)'] = pred
print(submission.head())

submission.to_csv(
    './dacon/_output/3_scientific/dacon_baseline.csv', index=False)

# from dacon_submit_api import dacon_submit_api

# result = dacon_submit_api.post_submission_file(
#     'dacon_baseline.csv',
#     '개인 Token',
#     '235789',
#     'DACONIO',
#     'DACON_Baseline'
# )
