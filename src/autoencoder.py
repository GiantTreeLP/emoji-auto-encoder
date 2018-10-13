import datetime
import glob
from os import path

import imageio
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.activations import relu, sigmoid
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.metrics import binary_accuracy
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adadelta

from image_callback import TensorBoardImage


def create_model() -> Model:
    input_img = Input(shape=(128, 128, 4))  # 128x128
    x = Conv2D(16, (3, 3), activation=relu, padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 64x64
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 32x32
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((4, 4), padding='same')(x)  # 8x8
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    encoded = Dense(8, activation=relu)(x)  # MaxPooling2D((4, 4), padding='same')(x)  # 2x2 = 4

    x = Conv2D(8, (3, 3), activation=relu, padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)  # 8x8
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 32x32
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 64x64
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 128x128
    decoded = Conv2D(4, (3, 3), activation=sigmoid, padding='same')(x)

    autoencoder = Model(input_img, decoded, name="emoji-autoencoder")
    autoencoder.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[binary_accuracy])
    return autoencoder


def train_model(model: Model, images):
    time_str = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    callbacks = [
        TensorBoard(log_dir=f'../logs/{time_str}', write_images=True),
        ProgbarLogger("samples", stateful_metrics="loss"),
        TensorBoardImage(f'../logs/{time_str}', "Emojis", images)
    ]
    model.fit(images, images, epochs=1000, shuffle=True, callbacks=callbacks)
    model.save("../logs/model.h5")


if __name__ == '__main__':
    images = []

    for file in glob.glob("../emojis/twemoji/png/*.png"):
        images.append(imageio.imread(file))

    images = np.array(images)
    images = images.astype('float32') / 255

    if path.exists("../logs/model.h5"):
        model: Model = load_model("../logs/model.h5")
    else:
        model = create_model()

    train_model(model, images)

    prediction = model.predict([[images[0]]])
    imageio.imwrite("original.png", images[0])
    imageio.imwrite("test.png", prediction[0])

    # np.reshape(images, 128, 128, 4)
    # main()
    pass
