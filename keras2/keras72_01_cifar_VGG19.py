# cifar10과 100
# trainable True/False
# Model FC/GAP 비교
# trainable 동결, 미동결 비교
# fc 모델, average pooling 비교

from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10, cifar100
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta, RMSprop, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping

COUNT = 1
LOSS_ACC_LS = []
DATASETS = {'cifar_10': cifar10.load_data(), 'cifar100': cifar100.load_data()}
TRAINABLE = {'True_': True, 'False': False}
FLATTEN_GAP = {'Flatten': Flatten(), 'GAP__2D': GlobalAveragePooling2D()}

for dt_key, dt_val in DATASETS.items():
    #1 Data
    (x_train, y_train), (x_test, y_test) = dt_val
    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)
    # ic(x_train.shape, x_test.shape)
    # ic(np.unique(y_train))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    #2 Model
    for tf_key, tf_val in TRAINABLE.items():
        for fg_key, fg_val in FLATTEN_GAP.items():
            vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
            vgg19.trainable = tf_val

            model = Sequential()
            model.add(vgg19)
            model.add(fg_val)
            if dt_key == 'cifar10':
                model.add(Dense(100, activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(10, activation='softmax'))
            else:
                model.add(Dense(200, activation='relu'))
                model.add(Dense(100, activation='softmax'))
            # model.summary()

            #3 Train
            opt = Adam()
            model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=opt, metrics=['acc'])
            es = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)
            model.fit(x_train, y_train, epochs=20, batch_size=512,
                    verbose=1, validation_split=0.25, callbacks=[es])

            #4 Evaluate
            loss = model.evaluate(x_test, y_test, batch_size=128)
            result = f'[{COUNT}] {dt_key}_{tf_key}_{fg_key} :: loss= {round(loss[0], 5)}, acc= {round(loss[1], 5)}'
            ic(result)
            LOSS_ACC_LS.append(result)
            COUNT = COUNT + 1

ic('================= RESULTS ==================')
for i in LOSS_ACC_LS:
    print(i)

'''
VGG19
[1] cifar10_True_Flatten :: loss= 0.859976053237915, acc= 0.7800999879837036
[2] cifar10_True_GAP__2D :: loss= 0.8568546772003174, acc= 0.7886000275611877
[3] cifar10_False_Flatten :: loss= 1.0596932172775269, acc= 0.641700029373169
[4] cifar10_False_GAP__2D :: loss= 1.0640227794647217, acc= 0.6380000114440918
[5] cifar100_True_Flatten :: loss= 3.2498319149017334, acc= 0.31940001249313354
[6] cifar100_True_GAP__2D :: loss= 3.306947946548462, acc= 0.30410000681877136
[7] cifar100_False_Flatten :: loss= 2.418550968170166, acc= 0.38519999384880066
[8] cifar100_False_GAP__2D :: loss= 2.3991663455963135, acc= 0.39259999990463257
'''