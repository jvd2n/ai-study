import time
import numpy as np
from icecream import ic
from tensorflow.keras.preprocessing.image import ImageDataGenerator

start_time = time.time()

train_idg = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2
)

test_idg = ImageDataGenerator(
    rescale=1./255,
)

train_gen = train_idg.flow_from_directory(
    '../data/guitar_chord',
    target_size=(150, 150),
    batch_size=11000,
    class_mode='categorical',
    subset='training',
)
# Found 10762 images belonging to 14 classes.

valid_gen = train_idg.flow_from_directory(
    '../data/guitar_chord',
    target_size=(150, 150),
    batch_size=2700,
    class_mode='categorical',
    subset='validation',
)
# Found 2681 images belonging to 14 classes.

test_gen = test_idg.flow_from_directory(
    '../data/guitar_chord_pred',
    batch_size=14,
    target_size=(150, 150),
    class_mode='categorical',
    shuffle=False,
)
# Found 14 images belonging to 14 classes.

ic(train_gen[0][0].shape)   # (10762, 150, 150, 3)
ic(train_gen[0][1].shape)   # (10762, 14)
ic(valid_gen[0][0].shape)   # (2681, 150, 150, 3)
ic(valid_gen[0][1].shape)   # (2681, 14)
ic(test_gen[0][0].shape)   # (14, 150, 150, 3)
ic(test_gen[0][1].shape)   # (14, 14)

# ic(train_gen.shape)
# ic(valid_gen.shape)
'''
augment_size = 50000 - train_gen[0][0].shape[0]

randidx = np.random.randint(train_gen[0][0].shape[0], size=augment_size)
ic(train_gen[0][0].shape[0])     # 60000
ic(randidx)                      # [48716 39008   951 ... 18997  4609 20790]
ic(randidx.shape)                # (40000,)

x_augmented = train_gen[0][0][randidx].copy()
y_augmented = train_gen[0][1][randidx].copy()

ic(x_augmented.shape)    # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 150, 150, 3)
x_train = train_gen[0][0].reshape(train_gen[0][0].shape[0], 150, 150, 3)

x_augmented = train_idg.flow(
    x_augmented,
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=False
).next()[0]

ic(x_augmented.shape)   # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((train_gen[0][1], y_augmented))

ic(x_train.shape, y_train.shape)    # (100000, 28, 28, 1), (100000,)
'''

np.save('./_save/_npy/ygc_train_x.npy', arr=train_gen[0][0])
# np.save('./_save/_npy/ygc_train_x.npy', arr=x_train)
np.save('./_save/_npy/ygc_train_y.npy', arr=train_gen[0][1])
# np.save('./_save/_npy/ygc_train_y.npy', arr=y_train)
np.save('./_save/_npy/ygc_valid_x.npy', arr=valid_gen[0][0])
np.save('./_save/_npy/ygc_valid_y.npy', arr=valid_gen[0][1])
np.save('./_save/_npy/ygc_test_x.npy', arr=test_gen[0][0])
np.save('./_save/_npy/ygc_test_y.npy', arr=test_gen[0][1])

duration_time = time.time() - start_time

ic(duration_time)