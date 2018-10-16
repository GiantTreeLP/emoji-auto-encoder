import datetime
import glob
import os
from os import path
from typing import Tuple

import imageio
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.activations import relu, tanh
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.optimizers import Adadelta

from checkpoint_callback import CheckpointCallback
from image_callback import TensorBoardImage

LOGS_DIR = "../logs/"


def encoder_128(vector_len: int) -> Model:
    input_img = Input(shape=(128, 128, 1), name="input_128x128")  # 128x128
    x = Conv2D(16, (5, 5), activation=relu, padding='same', name="Convolution1")(input_img)
    x = MaxPooling2D((2, 2), padding='same', name="shrink_64x64")(x)  # 64x64
    x = Conv2D(8, (3, 3), activation=relu, padding='same', name="Convolution2")(x)
    x = MaxPooling2D((2, 2), padding='same', name="shrink_32x32")(x)  # 32x32
    x = Conv2D(8, (3, 3), activation=relu, padding='same', name="Convolution3")(x)
    x = MaxPooling2D((4, 4), padding='same', name="shrink_8x8")(x)
    x = Conv2D(4, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same', name="shrink_4x4")(x)
    x = Flatten(name="matrix-to-vector")(x)
    x = Dense(64, activation=relu, name="link_flat_to_64x1")(x)
    encoded = Dense(vector_len, activation=tanh, name=f"output_{vector_len}x1")(x)
    return Model(input_img, encoded, name="Encoder")


def decoder_128(vector_len: int) -> Model:
    input_decoder = Input(shape=(vector_len,), name=f"input_{vector_len}x1")
    x = Dense(64, activation=tanh, name="activate_input")(input_decoder)
    x = Dense(64, activation=relu, name="link_reshape_64x1")(x)
    x = Reshape((8, 8, 1), name="reshape_8x8")(x)
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2), name="grow_16x16")(x)
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2), name="grow_32x32")(x)
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2), name="grow_64x64")(x)
    x = Conv2D(16, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2), name="grow_128x128")(x)
    decoded = Conv2D(1, (5, 5), activation=tanh, padding='same', name="output_128x128")(x)

    return Model(input_decoder, decoded, name="Decoder")


def create_model(vector_len: int) -> Tuple[Model, Model, Model]:
    encoder = encoder_128(vector_len)
    decoder = decoder_128(vector_len)

    input_layer = Input(shape=(128, 128, 1))

    autoencoder = Model(input_layer, decoder(encoder(input_layer)), name="emoji_autoencoder")
    autoencoder.compile(optimizer=Adadelta(), loss=mean_squared_error)
    return autoencoder, encoder, decoder


def train_model(model: Model, images):
    time_str = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    callbacks = [
        TensorBoardImage(f'{LOGS_DIR}{time_str}', "Emojis", images, period=10),
        CheckpointCallback(f'{LOGS_DIR}{time_str}', period=10),
    ]
    model.fit(images, images, epochs=100000, batch_size=len(images),
              # validation_data=(images, images),
              callbacks=callbacks,
              verbose=0)
    model.save(f"../logs/{time_str}/model.h5")


def get_model():
    model, encoder, decoder = create_model(8)
    if path.exists(LOGS_DIR):
        dirs = [x for x in os.listdir(LOGS_DIR) if not path.isfile(f"{LOGS_DIR}{x}")]
        dirs.sort()

        if path.exists(f"{LOGS_DIR}{dirs[-1]}/model.h5"):
            try:
                model.load_weights(f"{LOGS_DIR}{dirs[-1]}/model.h5")
            except ValueError:
                pass
    return model, encoder, decoder


def main():
    images = []
    for file in glob.glob("../emojis/twemoji/png_bw/*.png"):
        images.append(imageio.imread(file))
    images *= 2  # increase batch input by duplication
    images = np.array(images)
    images = np.reshape(images, (-1, 128, 128, 1))
    images = images.astype('float32') / 255
    model, _, _ = get_model()
    model.summary()
    train_model(model, images)


if __name__ == '__main__':
    main()
