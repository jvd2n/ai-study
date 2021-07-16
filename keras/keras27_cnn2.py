from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2),                    # (N, 10, 10, 1) 
                 padding='same', input_shape=(10, 10, 1)))  # (N, 10, 10, 10)
model.add(Conv2D(20, (2, 2), activation='relu'))            # (N, 9, 9, 20)
model.add(Conv2D(30, (2, 2), padding='valid'))              # (N, 8, 8, 30)
model.add(MaxPool2D())                                      # (N, 4, 4, 30)
model.add(Conv2D(15, (2,2)))
model.add(Flatten())                                        # (N, 480)
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#1 Conv2D의 디폴트 액티베이션
#2 Conv2D summary의 파라미터 개수 완벽 이해
