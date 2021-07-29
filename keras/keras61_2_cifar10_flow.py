import numpy as np 
from icecream import ic
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=.1,
    height_shift_range=.1,
    rotation_range=10,
    zoom_range=.5,
    shear_range=.5,
    fill_mode='nearest'
)

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
# )

# 1 ImageDataGenerator를 정의
# 2 파일에서 가져오려면 -> flow_from_directory()    // x, y가 tuple 형태로 뭉쳐있음
# 3 데이터에서 땡겨오려면 -> flow()                 // x, y가 나뉘어있음

augment_size = 100000 - x_train.shape[0]

randidx = np.random.randint(x_train.shape[0], size=augment_size)
ic(x_train.shape[0])     # ic| x_train.shape[0]: 50000000
ic(randidx)              # ic| randidx: array([20648, 10982,  9502, ..., 29041, 11683, 16800])
ic(randidx.shape)        # ic| randidx.shape: (50000,)0000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

ic(x_augmented.shape)    # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3] if x_train.shape[3] else 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3] if x_train.shape[3] else 1)
x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3] if x_train.shape[3] else 1)

x_augmented = train_datagen.flow(
    x_augmented, 
    np.zeros(augment_size),
    batch_size=augment_size, 
    shuffle=False,
    save_to_dir='d:/temp/'
) #.next()[0]

# ic(x_augmented.shape)   # (40000, 28, 28, 1)
print(x_augmented[0][0].shape)   # (40000, 28, 28, 1)
print(x_augmented[0][1].shape)   # (40000, 28, 28, 1)
print(x_augmented[0][1][:10])   # (40000, 28, 28, 1)
print(x_augmented[0][1][10:15])   # (40000, 28, 28, 1)
# iterator 구조로 인해 next가 있을 때는 해당 객체가 실행 될 때마다 save_to_dir이 실행 되는 것으로 보임

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

ic(x_train.shape, y_train.shape)    # (100000, 28, 28, 1), (100000,)