import glob
import os
from typing import Tuple

import imageio
import numpy as np
import tensorflow as tf
from keras import backend as K

from autoencoder import get_model


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_images() -> Tuple[np.ndarray, list]:
    images = []
    names = []
    for file in glob.glob("../emojis/twemoji/png_bw/*.png"):
        images.append(imageio.imread(file))
        names.append(os.path.splitext(os.path.basename(file))[0])
    images = np.array(images)
    images = np.reshape(images, (-1, 128, 128, 1))
    images = images.astype('float32') / 255
    return images, names


def main():
    os.makedirs("../test/", exist_ok=True)
    images, names = load_images()
    _, encoder, decoder = get_model(8)
    image = images[0]
    intermediate_output = None
    for i in range(0, len(encoder.layers)):
        os.makedirs(f"../test/encoder_{i}", exist_ok=True)
        model = K.function([encoder.layers[0].input, K.learning_phase()], [encoder.layers[i].output])
        intermediate_output = model([[image]])[0]
        if len(intermediate_output.shape) == 4:
            intermediate_output = np.swapaxes(intermediate_output, 0, 3)
            intermediate_output = (intermediate_output * 255).astype('uint8')
            for j in range(len(intermediate_output)):
                imageio.imwrite(f"../test/encoder_{i}/{j}.png", intermediate_output[j], "png")

    encoded = intermediate_output

    for i in range(1, len(decoder.layers)):
        os.makedirs(f"../test/decoder_{i}", exist_ok=True)
        model = K.function([decoder.layers[0].input, K.learning_phase()], [decoder.layers[i].output])
        intermediate_output = model([encoded])[0]
        if len(intermediate_output.shape) == 4:
            intermediate_output = np.swapaxes(intermediate_output, 0, 3)
            intermediate_output = (intermediate_output * 255).astype('uint8')
            for j in range(len(intermediate_output)):
                imageio.imwrite(f"../test/decoder_{i}/{j}.png", intermediate_output[j], "png")
        if len(intermediate_output.shape) == 2:
            pass


if __name__ == '__main__':
    main()
