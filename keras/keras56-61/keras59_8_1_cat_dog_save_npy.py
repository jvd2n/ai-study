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
    validation_split=0.2,
)

test_idg = ImageDataGenerator(
    rescale=1./255,
)

train_gen = train_idg.flow_from_directory(
    '../data/cat_dog/training_set',
    target_size=(150, 150),
    batch_size=6500,
    # class_mode='binary',
    class_mode='categorical',
    shuffle=True,
    subset='training',
)
# Found 6404 images belonging to 2 classes.

valid_gen = train_idg.flow_from_directory(
    '../data/cat_dog/training_set',
    target_size=(150, 150),
    batch_size=1700,
    # class_mode='binary',
    class_mode='categorical',
    shuffle=True,
    subset='validation',
)
# Found 1601 images belonging to 2 classes.

# valid_gen = valid_idg.flow_from_directory(
test_gen = test_idg.flow_from_directory(
    '../data/cat_dog/test_set',
    target_size=(150, 150),
    batch_size=2100,
    shuffle=True,
    # class_mode='binary',
    class_mode='categorical',
)
# Found 2023 images belonging to 2 classes.

ic(train_gen[0][0].shape)   # (6404, 150, 150, 3)
ic(train_gen[0][1].shape)   # (6404,)
ic(valid_gen[0][0].shape)   # (1601, 150, 150, 3)
ic(valid_gen[0][1].shape)   # (1601,)
ic(test_gen[0][0].shape)   # (2023, 150, 150, 3)
ic(test_gen[0][1].shape)   # (2023,)

np.save('./_save/_npy/k59_9_train_x.npy', arr=train_gen[0][0])
np.save('./_save/_npy/k59_9_train_y.npy', arr=train_gen[0][1])
np.save('./_save/_npy/k59_9_valid_x.npy', arr=valid_gen[0][0])
np.save('./_save/_npy/k59_9_valid_y.npy', arr=valid_gen[0][1])
np.save('./_save/_npy/k59_9_test_x.npy', arr=test_gen[0][0])
np.save('./_save/_npy/k59_9_test_y.npy', arr=test_gen[0][1])

duration_time = time.time() - start_time

ic(duration_time)