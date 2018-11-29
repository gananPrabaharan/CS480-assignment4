
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, ZeroPadding2D
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
import PIL


def VG11():
    model = Sequential()
    # Input 28x28
    model.add(ZeroPadding2D(padding=((2, 2), (2, 2)), input_shape=(32, 32, 3)))

    # 32x32x3
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # 32x32
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # 16x16
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add((Activation('relu')))
    # 16x16
    model.add(MaxPooling2D(pool_size=(2, 2), stride=2))
    # 8x8
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add((Activation('relu')))
    # 8x8
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add((Activation('relu')))
    # 8x8
    # model.add(MaxPooling2D(pool_size=(2, 2), stride=2))
    model.add(MaxPooling2D(pool_size=(2, 2), stride=1))
    # 7x7
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add((Activation('relu')))
    # 7x7
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add((Activation('relu')))
    # 7x7
    # model.add(MaxPooling2D(pool_size=(2, 2), stride=2))
    model.add(MaxPooling2D(pool_size=(2, 2), stride=1))
    # 6x6
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add((Activation('relu')))
    # 6x6
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add((Activation('relu')))
    # 6x6
    model.add(MaxPooling2D(pool_size=(2, 2), stride=2))
    # 3x3
    model.add(Flatten())

    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dense(1000))
    model.add(Activation('relu'))

    model.add(Activation('softmax'))

    sgd_opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd_opt, metrics='accuracy')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    model.fit(x_train, y_train, epochs=1, batch_size=64)

VG11()
