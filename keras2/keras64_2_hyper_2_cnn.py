# CNN으로 변경
# 파라미터 변경
# 노드의 개수, activation도 추가
# epochs = [1, 2, 3]
# learning_rate 추가

# 나중엔 layer도 파라미터로 만들어보기 
# Dense, Conv 등

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Conv1D, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta, RMSprop, SGD, Nadam

#1 Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28*28, 1).astype('float32')/255
x_test = x_test.reshape(-1, 28*28, 1).astype('float32')/255

print(x_train.shape, x_test.shape)

#2 Model
def build_model(drop=0.5, optimizer='adam', node=64, activation='linear', lr=0.1):
    model = Sequential()
    model.add(Conv1D(filters=node, kernel_size=2, padding='same', activation=activation, input_shape=(28*28, 1)))
    model.add(Dropout(drop))
    model.add(Conv1D(node, 2, padding='same', activation=activation))
    model.add(MaxPool1D())
    model.add(Dropout(drop))
    model.add(Conv1D(node/2, 2, padding='same', activation=activation))
    model.add(MaxPool1D())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(10, activation='softmax'))
    opt = optimizer(lr=lr)
    model.compile(optimizer=opt, metrics=['acc'],
                  loss='categorical_crossentropy')
    
    # input = Input((28, 28))
    # xx = Conv1D(node, 2, activation=activation)(input)
    # xx = MaxPool1D()(xx)
    # xx = Conv1D(node, 2, activation=activation)(xx)
    # xx = Dropout(drop)(xx)
    # xx = Conv1D(node/2, 2, activation=activation)(xx)
    # xx = GlobalAveragePooling1D()(xx)
    # output = Dense(10, activation='softmax')(xx)
    # model = Model(inputs=input, outputs=output)
    '''
    from keras.layers import Conv2D, Input

    # 3-채널 256x256 이미지에 대한 인풋 텐서
    x = Input(shape=(256, 256, 3))
    # 3개의 아웃풋 채널(인풋 채널과 동일)을 가진 3x3 컨볼루션
    y = Conv2D(3, (3, 3), padding='same')(x)
    # 이는 x + y를 반환합니다
    z = keras.layers.add([x, y])
    '''
    # inputs = Input(shape=(28*28, 1), name='inputs')
    # x = Dense(512, activation='relu', name='hidden1')(inputs)
    # x = Dropout(drop)(x)
    # x = Dense(256, activation='relu', name='hidden2')(x)
    # x = Dropout(drop)(x)
    # x = Dense(128, activation='relu', name='hidden3')(x)
    # x = Dropout(drop)(x)
    # outputs = Dense(10, activation='softmax', name='outputs')(x)
    # model = Model(inputs=inputs, outputs=outputs)
    # model.compile(optimizer=optimizer, metrics=['acc'],
    #               loss='categorical_crossentropy')
    return model

def create_hyper_parameter():
    batches = [1000, 2000, 3000, 4000, 5000]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.3, 0.4, 0.5]
    epochs = [1, 2, 3]
    activations = ['relu', 'selu']
    nodes = [256, 128]
    learning_rate = [0.1, 0.01, 0.001]
    return {
        'batch_size': batches, 
        'optimizer': optimizers,
        'drop': dropouts, 
        'epochs': epochs, 
        'activation': activations, 
        'node': nodes,
        'lr': learning_rate, 
    }

hyper_parameter = create_hyper_parameter()
# print(hyper_parameter)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1) #, validation_split=0.2) #, epochs=2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model = RandomizedSearchCV(model2, hyper_parameter, cv=5)
model = GridSearchCV(model2, hyper_parameter, cv=2)

model.fit(x_train, y_train, verbose=1, epochs=2, validation_split=0.2)

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print('최종 스코어: ', acc)