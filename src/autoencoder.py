import datetime
import glob
from os import path
from typing import Tuple

import imageio
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.activations import relu, sigmoid
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.optimizers import Adadelta

from checkpoint_callback import CheckpointCallback
from image_callback import TensorBoardImage


def encoder_128(vector_len: int) -> Model:
    input_img = Input(shape=(128, 128, 1), name="input 128x128")  # 128x128
    x = Conv2D(16, (3, 3), activation=relu, padding='same', name="Convolution1")(input_img)
    x = MaxPooling2D((2, 2), padding='same', name="shrink 64x64")(x)  # 64x64
    x = Conv2D(8, (3, 3), activation=relu, padding='same', name="Convolution2")(x)
    x = MaxPooling2D((2, 2), padding='same', name="shrink 32x32")(x)  # 32x32
    x = Conv2D(8, (3, 3), activation=relu, padding='same', name="Convolution3")(x)
    x = MaxPooling2D((4, 4), padding='same', name="shrink 8x8")(x)
    x = Conv2D(4, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same', name="shrink 4x4")(x)
    x = Flatten(name="matrix to vector")(x)
    x = Dense(64, activation=relu, name="link flat to 64x1")(x)
    encoded = Dense(vector_len, activation=relu, name=f"output {vector_len}x1")(x)
    return Model(input_img, encoded, name="Encoder")


def decoder_128(vector_len: int) -> Model:
    input_decoder = Input(shape=(vector_len,), name=f"input {vector_len}x1")
    x = Dense(64, activation=relu, name="link reshape 64x1")(input_decoder)
    x = Reshape((8, 8, 1), name="reshape 8x8")(x)
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2), name="grow 16x16")(x)
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2), name="grow 32x32")(x)
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2), name="grow 64x64")(x)
    x = Conv2D(16, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2), name="grow 128x128")(x)
    decoded = Conv2D(1, (3, 3), activation=sigmoid, padding='same', name="output 128x128")(x)

    return Model(input_decoder, decoded, name="Decoder")


def create_model(vector_len: int) -> Tuple[Model, Model, Model]:
    encoder = encoder_128(vector_len)
    decoder = decoder_128(vector_len)

    input_layer = Input(shape=(128, 128, 1))

    autoencoder = Model(input_layer, decoder(encoder(input_layer)), name="emoji-autoencoder")
    autoencoder.compile(optimizer=Adadelta(), loss=mean_squared_error)
    return autoencoder, encoder, decoder


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
    model.fit(images, images, epochs=100000 + epoch, batch_size=len(images),
              # validation_data=(images, images),
              initial_epoch=epoch,
              callbacks=callbacks,
              verbose=0)
    model.save("../logs/model.h5")


def get_model():
    global model
    model, _, _ = create_model(64)
    if path.exists("../logs/model.h5"):
        model.load_weights("../logs/model.h5")
    return model


if __name__ == '__main__':
    images = []

    for file in glob.glob("../emojis/twemoji/png_bw/*.png"):
        images.append(imageio.imread(file))

    images = np.array(images)
    images = np.reshape(images, (-1, 128, 128, 1))
    images = images.astype('float32') / 255

    model = get_model()

    model.summary()

    train_model(model, images)
