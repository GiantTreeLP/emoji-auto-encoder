import datetime
import glob
import os
from os import path
from typing import Tuple

import imageio.v3 as imageio
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.activations import relu, tanh, softplus
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, MaxPooling2D
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.optimizer_v2.adadelta import Adadelta

from checkpoint_callback import CheckpointCallback
from image_callback import TensorBoardImage

LOGS_DIR = "../logs/"


def encoder_128(vector_len: int) -> Model:
    input_img = Input(shape=(128, 128, 4), name="Input_128x128")  # 128x128
    x = Conv2D(64, (5, 5), strides=2, activation=relu, padding='same', name="Convolution_64x64")(input_img)
    x = Conv2D(32, (5, 5), strides=2, activation=relu, padding='same', name="Convolution_32x32")(x)
    x = Conv2D(16, (3, 3), strides=2, activation=relu, padding='same', name="Convolution_16x16")(x)
    x = Conv2D(8, (3, 3), strides=2, activation=relu, padding='same', name="Convolution_8x8")(x)
    x = Flatten(name="Matrix_To_Vector")(x)
    encoded = Dense(vector_len, activation=tanh, name=f"Output_{vector_len}x1")(x)
    return Model(input_img, encoded, name="Encoder")


def decoder_128(vector_len: int) -> Model:
    input_decoder = Input(shape=(vector_len,), name=f"input_{vector_len}x1")
    x = Dense(64, activation=tanh, name="Activate_Input")(input_decoder)
    x = Reshape((8, 8, 1), name="Reshape_8x8")(x)
    x = Conv2DTranspose(16, (3, 3), strides=1, activation=relu, padding='same', name="Transpose_8x8")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, activation=relu, padding='same', name="Transpose_16x16")(x)
    x = Conv2DTranspose(32, (5, 5), strides=2, activation=relu, padding='same', name="Transpose_32x32")(x)
    x = Conv2DTranspose(64, (5, 5), strides=2, activation=relu, padding='same', name="Transpose_64x64")(x)
    decoded = Conv2DTranspose(4, (5, 5), strides=2, activation=sigmoid, padding='same', name="Output_128x128")(x)

    return Model(input_decoder, decoded, name="Decoder")


def create_model(vector_len: int) -> Tuple[Model, Model, Model]:
    encoder = encoder_128(vector_len)
    decoder = decoder_128(vector_len)

    input_layer = Input(shape=(128, 128, 4))

    autoencoder = Model(input_layer, decoder(encoder(input_layer)), name="emoji_autoencoder")
    autoencoder.compile(optimizer=Adadelta(learning_rate=1.0, epsilon=1e-6), loss=mean_squared_error)
    return autoencoder, encoder, decoder


def train_model(model: Model, images: list[np.ndarray]):
    time_str = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    train_ds = Dataset.from_tensor_slices((images, images)).shuffle(len(images)).cache().prefetch(
        buffer_size=AUTOTUNE).batch(len(images))

    callbacks = [
        TensorBoard(f'{LOGS_DIR}{time_str}'),
        TensorBoardImage(f'{LOGS_DIR}{time_str}', "Emojis", images, period=100),
        CheckpointCallback(f'{LOGS_DIR}{time_str}', period=100),
    ]
    model.fit(train_ds, epochs=100000, batch_size=len(images),
              # validation_data=(images, images),
              callbacks=callbacks,
              verbose=0)
    model.save(f"../logs/{time_str}/model.h5")


def get_model(vector_len):
    model, encoder, decoder = create_model(vector_len)
    if path.exists(LOGS_DIR):
        dirs = [x for x in os.listdir(LOGS_DIR) if not path.isfile(f"{LOGS_DIR}{x}")]
        dirs.sort()

        if dirs:
            model_dir = dirs[-1]
        else:
            return model, encoder, decoder

        if path.exists(f"{LOGS_DIR}{model_dir}/model"):
            try:
                print(f"Loading model '{model_dir}'")
                model.load_weights(f"{LOGS_DIR}{model_dir}/model")
            except ValueError:
                pass
    return model, encoder, decoder


def main():
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    images = []
    for file in glob.glob("../emojis/twemoji/png/*.png"):
        images.append(imageio.imread(file))
    images = np.array(images)
    images = np.reshape(images, (-1, 128, 128, 4))
    images = images.astype('float32') / 255
    model, encoder, decoder = get_model(16)
    encoder.summary()
    decoder.summary()
    model.summary()
    train_model(model, images)


if __name__ == '__main__':
    main()
