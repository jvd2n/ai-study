import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000,) (10000,)
x = np.append(x_train, x_test, axis=0)
print(x.shape)  # (70000, 28, 28)

# pca를 통해 0.95 이상인 n_components가 몇 개?
# Modeling
# Tensorflow DNN으로 구성하고 기존 DNN과 비교

N_COMPONENTS=154
pca = PCA(n_components=N_COMPONENTS)   # 컬럼을 n_components개로 압축
# x_train = pca.fit_transform(x_train.reshape(-1, 28*28))
# x_test = pca.fit_transform(x_test.reshape(-1, 28*28))
x = pca.fit_transform(x.reshape(-1, 28*28))
print(x.shape)
x_train = x[:60000]
x_test = x[-10000:]
print(x_train.shape, x_test.shape)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

# print(np.argmax(cumsum >= 0.95) + 1)

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()


# Data preprocessing
# x_train = x_train.reshape(-1, N_COMPONENTS)
# x_test = x_test.reshape(-1, N_COMPONENTS)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, PowerTransformer
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling1D

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(N_COMPONENTS,)))
# model.add(Dense(100, activation='relu', input_shape=(28 * 28, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(GlobalAveragePooling1D())
model.add(Dense(10, activation='softmax'))

# DNN 구해서 CNN 비교
# DNN + GlobalAveragePooling 구해서 CNN 비교

#3 Compile, Train   metrics=['accuracy']
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

model.fit(
    x_train, y_train, 
    epochs=30, 
    batch_size=64, 
    verbose=1,
    validation_split=0.2, 
    callbacks=[es]
)

#4 Evaluate
from sklearn.metrics import r2_score, accuracy_score
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
# results = model.score(x_test, y_test)
# y_pred = model.predict(x_test)
print(f'loss: {loss[0]}')
print(f'accuracy: {loss[1]}')
# acc = accuracy_score(y_test, y_pred)
# print('acc_score : ', acc)
# print(results)
'''
loss: 0.18337686359882355
accuracy: 0.9632999897003174
'''