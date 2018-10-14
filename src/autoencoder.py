import datetime
import glob
from os import path

import imageio
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.activations import relu, sigmoid
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.metrics import binary_accuracy
from tensorflow.python.keras.optimizers import Adadelta

from checkpoint_callback import CheckpointCallback
from image_callback import TensorBoardImage


def create_model(vector_len: int) -> Model:
    input_img = Input(shape=(128, 128, 1), name="Input-Image-128x128")  # 128x128
    x = Conv2D(16, (3, 3), activation=relu, padding='same', name="Convolution1")(input_img)
    x = MaxPooling2D((2, 2), padding='same', name="64x64")(x)  # 64x64
    x = Conv2D(8, (3, 3), activation=relu, padding='same', name="Convolution2")(x)
    x = MaxPooling2D((2, 2), padding='same', name="32x32")(x)  # 32x32
    x = Conv2D(8, (3, 3), activation=relu, padding='same', name="Convolution3")(x)
    x = MaxPooling2D((4, 4), padding='same', name="8x8")(x)  # 8x8
    x = Conv2D(4, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 2x2x4 = 16
    x = Flatten()(x)
    x = Dense(64, activation=relu)(x)
    encoded = Dense(vector_len, activation=relu)(x)

    # encoder = Model(input_img, encoded, name="Encoder")

    input_decoder = Dense(vector_len, activation=relu)(encoded)
    x = Dense(16, activation=relu)(input_decoder)
    x = Reshape((4, 4, 1))(x)
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 8x8
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 32x32
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 64x64
    x = Conv2D(16, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((4, 4))(x)  # 128x128
    decoded = Conv2D(1, (3, 3), activation=sigmoid, padding='same', name="Decoded")(x)

    # decoder = Model(input_decoder, decoded, name="Decoder")

    autoencoder = Model(input_img, decoded, name="emoji-autoencoder")
    autoencoder.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[binary_accuracy])
    return autoencoder


def train_model(model: Model, images):
    time_str = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    if path.exists("../logs/epoch"):
        epoch = int(open("../logs/epoch", "r").readline())
    else:
        epoch = 0

    callbacks = [
        TensorBoardImage(f'../logs/{time_str}', "Emojis", images),
        CheckpointCallback("../logs/model.h5", period=100),
    ]
    model.fit(images, images, epochs=20000, batch_size=len(images),
              # validation_data=(images, images),
              initial_epoch=epoch,
              callbacks=callbacks)
    model.save("../logs/model.h5")


if __name__ == '__main__':
    images = []

    for file in glob.glob("../emojis/twemoji/png_bw/*.png"):
        images.append(imageio.imread(file))

    images = np.array(images)
    images = np.reshape(images, (-1, 128, 128, 1))
    images = images.astype('float32') / 255

    model = create_model(64)

    if path.exists("../logs/model.h5"):
        model.load_weights("../logs/model.h5")

    model.summary()

    train_model(model, images)
