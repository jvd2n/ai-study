import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D

#1 Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28*28).astype('float32')/255
x_test = x_test.reshape(-1, 28*28).astype('float32')/255

#2 Model
def build_model(drop=0.5, optimizer='adam'):
# def build_model(drop, optimizer):
    inputs = Input(shape=(28*28), name='inputs')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyper_parameter():
    batches = [1000, 2000, 3000, 4000, 5000]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.3, 0.4, 0.5]
    return {'batch_size': batches, 'optimizer': optimizers,
            'drop': dropouts}

hyper_parameter = create_hyper_parameter()
# print(hyper_parameter)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1) #, validation_split=0.2) #, epochs=2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model = RandomizedSearchCV(model2, hyper_parameter, cv=5)
model = GridSearchCV(model2, hyper_parameter, cv=2)

model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print('최종 스코어: ', acc)