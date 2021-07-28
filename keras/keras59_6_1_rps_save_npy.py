
import numpy as np
from icecream import ic
from tensorflow.keras.preprocessing.image import ImageDataGenerator

imageGen = ImageDataGenerator(
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
# test_datagen = ImageDataGenerator(rescale=1./255)

trainGen = imageGen.flow_from_directory(
    '../data/rps',
    target_size=(150, 150),
    batch_size=2100,
    class_mode='categorical',
    subset='training',
)
# Found 2648 images belonging to 2 classes.

testGen = imageGen.flow_from_directory(
    '../data/rps',
    target_size=(150, 150),
    batch_size=530,
    class_mode='categorical',
    subset='validation',
)
# Found 661 images belonging to 2 classes.

ic(trainGen[0][0].shape)
ic(trainGen[0][1].shape)
ic(testGen[0][0].shape)
ic(testGen[0][1].shape)

# ic(trainGen.shape)
# ic(testGen.shape)

np.save('./_save/_npy/k59_7_train_x.npy', arr=trainGen[0][0])
np.save('./_save/_npy/k59_7_train_y.npy', arr=trainGen[0][1])
np.save('./_save/_npy/k59_7_test_x.npy', arr=testGen[0][0])
np.save('./_save/_npy/k59_7_test_y.npy', arr=testGen[0][1])