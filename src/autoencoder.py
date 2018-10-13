from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.optimizers import Adadelta


def main():
    input_img = Input(shape=(128, 128, 3))  # 128x128
    x = Conv2D(16, (3, 3), activation=relu, padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 64x64
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 32x32
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((4, 4), padding='same')(x)  # 8x8
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    encoded = MaxPooling2D((4, 4), padding='same')(x)  # 2x2 = 4

    x = Conv2D(8, (3, 3), activation=relu, padding='same')(encoded)
    x = UpSampling2D((4, 4))(x)  # 8x8
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((4, 4))(x)  # 32x32
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 64x64
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 128x128
    decoded = Conv2D(16, (3, 3), activation=relu, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adadelta, loss=binary_crossentropy)


if __name__ == '__main__':
    # im = imread("../emojis/emojione/png/1F9D0.png")
    # print(im.shape)
    # np.reshape(images, 128, 128, 3)
    # main()
    pass
