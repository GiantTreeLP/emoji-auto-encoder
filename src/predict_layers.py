import glob
import os
from typing import Tuple

import imageio.v3 as imageio
import numpy as np
from tensorflow.python.keras import Model

from autoencoder import get_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_images() -> Tuple[np.ndarray, list]:
    images = []
    names = []
    for file in glob.glob("../emojis/twemoji/png/*.png"):
        images.append(imageio.imread(file))
        names.append(os.path.splitext(os.path.basename(file))[0])
    images = np.array(images)
    images = np.reshape(images, (len(images), -1, 128, 128, 4))
    images = images.astype('float32') / 255
    return images, names


def main():
    os.makedirs("../test/", exist_ok=True)
    images, names = load_images()
    _, encoder, decoder = get_model(16)
    image = images[76]

    for i in range(len(encoder.layers)):
        new_model = Model(inputs=encoder.input, outputs=encoder.layers[i].output)
        intermediate_images = (new_model.predict(image) * 255).astype('uint8')
        if len(intermediate_images.shape) == 4:
            intermediate_images = np.swapaxes(intermediate_images, 0, 3)
            intermediate_images = intermediate_images.reshape(intermediate_images.shape[:3])

            os.makedirs(f"../test/encoder_{i}", exist_ok=True)

            for j in range(len(intermediate_images)):
                im = intermediate_images[j]
                imageio.imwrite(f"../test/encoder_{i}/{j}.png", im)

    value = encoder.predict(image)

    for i in range(len(decoder.layers)):
        new_model = Model(inputs=decoder.input, outputs=decoder.layers[i].output)
        intermediate_images = (new_model.predict(value) * 255).astype('uint8')
        if len(intermediate_images.shape) == 4:
            intermediate_images = np.swapaxes(intermediate_images, 0, 3)
            intermediate_images = intermediate_images.reshape(intermediate_images.shape[:3])

            os.makedirs(f"../test/decoder_{i}", exist_ok=True)

            for j in range(len(intermediate_images)):
                im = intermediate_images[j]
                imageio.imwrite(f"../test/decoder_{i}/{j}.png", im)


if __name__ == '__main__':
    main()
