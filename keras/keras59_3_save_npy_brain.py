import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../data/brain/train',
    target_size=(150, 150),
    batch_size=200,   # xy_train[0]의 ,5(batch_size) 크기로 생성
    class_mode='binary'
)
# Found 160 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    '../data/brain/test',
    target_size=(150, 150),
    batch_size=200,   # xy_train[0]의 ,5(batch_size) 크기로 생성
    class_mode='binary',
    shuffle=True
) 
# Found 120 images belonging to 2 classes.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000002C3A9DB9780>
print(xy_train[0][0])   # x value
print(xy_train[0][1])   # y value
# print(xy_train[0][2])   # None
print(xy_train[0][0].shape, xy_train[0][1].shape)   # (160, 150, 150, 3) (160,)
print(xy_test[0][0].shape, xy_test[0][1].shape)     # (120, 150, 150, 3) (120,)

# print(xy_train[31][1])  # 마지막 배치 y. 총 32장 * batchsize = 160장의 사진임을 알 수 있다.
# print(xy_train[32][1])    # None

# print(type(xy_train))       # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))    # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

np.save('./_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
np.save('./_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
np.save('./_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
np.save('./_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])