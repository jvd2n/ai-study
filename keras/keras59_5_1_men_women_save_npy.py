import numpy as np
from icecream import ic
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

valid_idg = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

test_idg = ImageDataGenerator(rescale=1./255)

train_gen = train_idg.flow_from_directory(
    '../data/men_women',
    target_size=(150, 150),
    batch_size=3000,
    class_mode='binary',
    subset='training',
)
# Found 2648 images belonging to 2 classes.

valid_gen = valid_idg.flow_from_directory(
    '../data/men_women',
    target_size=(150, 150),
    batch_size=1000,
    class_mode='binary',
    subset='validation',
)
# Found 661 images belonging to 2 classes.

test_gen = test_idg.flow_from_directory(
    '../data/men_women_pred',
    batch_size=3,
    target_size=(150, 150),
    class_mode='binary',
)
# Found 3 images belonging to 1 classes.

ic(train_gen[0][0].shape)   # (2648, 150, 150, 3)
ic(train_gen[0][1].shape)   # (2648,)
ic(valid_gen[0][0].shape)   # (661, 150, 150, 3)
ic(valid_gen[0][1].shape)   # (661,)
ic(test_gen[0][0].shape)    # (3, 150, 150, 3)
ic(test_gen[0][1].shape)    # (3,)

np.save('./_save/_npy/k59_5_train_x.npy', arr=train_gen[0][0])
np.save('./_save/_npy/k59_5_train_y.npy', arr=train_gen[0][1])
np.save('./_save/_npy/k59_5_valid_x.npy', arr=valid_gen[0][0])
np.save('./_save/_npy/k59_5_valid_y.npy', arr=valid_gen[0][1])
np.save('./_save/_npy/k59_5_test_x.npy', arr=test_gen[0][0])
np.save('./_save/_npy/k59_5_test_y.npy', arr=test_gen[0][1])
