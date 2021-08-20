from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19, Xception, ResNet50, ResNet101, InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
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
    # x_train = x_train.reshape(-1, 32 * 32 * 3)
    # x_test = x_test.reshape(-1, 32 * 32 * 3)
    # # ic(x_train.shape, x_test.shape)
    # # ic(np.unique(y_train))

    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    # x_train = x_train.reshape(-1, 32, 32, 3)
    # x_test = x_test.reshape(-1, 32, 32, 3)
    
    x_train = tf.image.resize(
        x_train, [75, 75], method='nearest', preserve_aspect_ratio=False,
        antialias=False, name=None
    )
    x_test = tf.image.resize(
        x_test, [75, 75], method='nearest', preserve_aspect_ratio=False,
        antialias=False, name=None
    )
    
    #2 Model
    for tf_key, tf_val in TRAINABLE.items():
        for fg_key, fg_val in FLATTEN_GAP.items():
            transfer_learning = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
            transfer_learning.trainable = tf_val

            model = Sequential()
            model.add(transfer_learning)
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
            result = f'[{COUNT}] {dt_key}_{tf_key}_{fg_key} :: loss= {round(loss[0], 4)}, acc= {round(loss[1], 4)}'
            ic(result)
            LOSS_ACC_LS.append(result)
            COUNT = COUNT + 1

print('InceptionResNetV2')
for i in LOSS_ACC_LS:
    print(i)

'''
InceptionResNetV2
[1] cifar_10_True__Flatten :: loss= 0.5947, acc= 0.8794
[2] cifar_10_True__GAP__2D :: loss= 0.6643, acc= 0.8436
[3] cifar_10_False_Flatten :: loss= 43.2382, acc= 0.1
[4] cifar_10_False_GAP__2D :: loss= 3.6945, acc= 0.1
[5] cifar100_True__Flatten :: loss= 3.5605, acc= 0.3108
[6] cifar100_True__GAP__2D :: loss= 1.9821, acc= 0.6121
[7] cifar100_False_Flatten :: loss= 4.6053, acc= 0.01
[8] cifar100_False_GAP__2D :: loss= 4.6053, acc= 0.01
'''