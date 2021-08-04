# np.save('./_save/_npy/ygc_train_x.npy', arr=train_gen[0][0])
# np.save('./_save/_npy/ygc_train_y.npy', arr=train_gen[0][1])
# np.save('./_save/_npy/ygc_valid_x.npy', arr=valid_gen[0][0])
# np.save('./_save/_npy/ygc_valid_y.npy', arr=valid_gen[0][1])
# np.save('./_save/_npy/ygc_test_x.npy', arr=test_gen[0][0])
# np.save('./_save/_npy/ygc_test_y.npy', arr=test_gen[0][1])

import time
import numpy as np
from icecream import ic
x_train = np.load('./_save/_npy/ygc_train_x.npy')
y_train = np.load('./_save/_npy/ygc_train_y.npy')
x_valid = np.load('./_save/_npy/ygc_valid_x.npy')
y_valid = np.load('./_save/_npy/ygc_valid_y.npy')
x_test = np.load('./_save/_npy/ygc_test_x.npy')
y_test = np.load('./_save/_npy/ygc_test_y.npy')

ic(x_train.shape)    # (10762, 150, 150, 3)
ic(y_train.shape)    # (10762, 14)
ic(x_valid.shape)    # (2681, 150, 150, 3)
ic(y_valid.shape)    # (2681, 14)
ic(x_test.shape)     # (14, 150, 150, 3)
ic(y_test.shape)     # (14, 14)

ic(y_train[:10])

# ic(x_train[1])
# ic(x_test[1])
# ic(y_train[1])
# ic(y_test[1])

x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
x_valid = x_valid.reshape(-1, x_valid.shape[1] * x_valid.shape[2] * x_valid.shape[3])

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
ic(x_train)
ic(x_valid)

x_train = x_train.reshape(-1, 130, 130, 3)
x_valid = x_valid.reshape(-1, 130, 130, 3)

# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

model = Sequential()
model.add(InputLayer(input_shape=(130, 130, 3)))
model.add(Conv2D(16, (3, 3), (1, 1), 'same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(32, (3, 3), (1, 1), 'same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
# model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))


# VGG16_MODEL = VGG16(
#     input_shape = (150, 150, 3),
#     include_top = False,
#     weights = 'imagenet',
# )
# VGG16_MODEL.trainable = False
# # flatten이 없음 ( globalaveragepooling으로 대체 )
# #  ==> 가중치가 필요없음
# global_average_layer = GlobalAveragePooling2D()
# # FFNN의 가중치는 학습됨
# prediction_layer = Dense(7, activation ='softmax')
# model = Sequential([
#     VGG16_MODEL,
#     global_average_layer,
#     prediction_layer
# ])


'''
from tensorflow.keras.applications import EfficientNetB0
# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = EfficientNetB0(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(150, 150, 3)))
    
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(5, activation="relu")(headModel)
headModel = Dropout(0.8)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# # place the head FC model on top of the base model (this will become
# # the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# model.summary()

for layer in baseModel.layers:
	layer.trainable = False
    

print("[INFO] compiling model…")
# opt = Adam(lr=INIT_LR)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
	metrics=["acc"])
'''

model.summary()

# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['acc'],
    # options=run_opts,
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    mode='min', 
    verbose=1, 
    restore_best_weights=True,
)

#########################################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%y%m%d_%H%M")

filepath = './PROJ_your_guitar/_save/ModelCheckPoint/' + date_time + '/'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "YGC_", date_time, "_", filename])
#########################################################################

mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='min', 
    verbose=2, 
    save_best_only=True, 
    filepath=modelpath
)

model.save(filepath + 'YGC_MCP.h5')

start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=30,
    verbose=2,
    # batch_size=2,
    steps_per_epoch=4,
    validation_data=(x_valid, y_valid),
    validation_steps=8,
    callbacks=[es, mcp],
)
end_time = time.time() - start_time

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

ic(end_time)
ic(acc[-1])
ic(val_acc[-1])
ic(loss[-1])
ic(val_loss[-1])

y_pred = model.predict(x_test)
ic(y_pred)
ic(y_test)
y_pred = model.predict(x_test)
# ic(np.argmax(y_pred, axis=1))
# ic(np.argmax(y_pred, axis=1).shape)
# ic(np.argmax(y_test, axis=1).shape)

# print(f'Correct_Answer_Score: {(np.count_nonzero(results == True) / results.size) * 100}')
results = []
for i, j in enumerate(y_pred):
    results.append(np.argmax(y_pred[i]) == np.argmax(y_test[i]))
    ic(np.argmax(y_pred[i]), np.argmax(y_test[i]))
ic(results)

t_res = 0
for result in results:
    if result == True:
        t_res = t_res + 1
correct_answer_rate = round((t_res / len(results) * 100), 2)
print(correct_answer_rate, '%')

# for i, j in enumerate(y_pred):
#     if j >= 0.5:
#         chance = round(j[0]*100, 1)
#         print(f'{i+1}. Classified/ WOMEN with {chance}%')
#     else:
#         chance = round((1-j[0])*100, 1)
#         print(f'{i+1}. Classified/ MEN with {chance}%')

# ic(y_pred)

import matplotlib.pyplot as plt

plt.figure(figsize=(13,6))
plt.rc('font', family='NanumGothic')
# print(plt.rcParams['font.family'])

plt.subplot(121)
plt.plot(hist.history['acc'], 'r')
plt.plot(hist.history['val_acc'], 'c')
plt.title('acc, val_acc')
plt.xlabel('epochs')
plt.ylabel('acc, val_acc')
plt.legend(['train_acc', 'val_acc'])

plt.subplot(122)
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'c')
plt.title('loss, val_loss')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['train_loss', 'val_loss'])

plt.show()

'''
ic| acc[-1]: 0.8217522501945496
ic| val_acc[-1]: 0.6096823215484619
ic| loss[-1]: 0.37846890091896057
ic| val_loss[-1]: 0.8057641386985779
ic| y_pred: array([[0.5404332 ],
                   [0.47137678],
                   [0.04922736]], dtype=float32)
1. Classified/ WOMEN with 54.0%
2. Classified/ MEN with 52.9%
3. Classified/ MEN with 95.1%
'''
'''
pred_x = []
from tensorflow.keras.preprocessing import image
test_image = image.load_img('../data/men_women/pred/00.jpg', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

ic(result)
ic(result.argmax())
ic(result.shape)
'''