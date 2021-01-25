import glob
import os
from typing import Tuple

import imageio
import numpy as np
from keract import get_activations

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
    whole_model, encoder, decoder = get_model(16)
    image = images[76]

    intermediate_output = get_activations(encoder, image)

    for i, (key, output) in enumerate(intermediate_output.items(), start=1):
        intermediate_images = (output * 255).astype('uint8')
        if len(output.shape) == 4:
            intermediate_images = np.swapaxes(intermediate_images, 0, 3)
            for j in range(len(intermediate_images)):
                im = intermediate_images[j]
                imageio.imwrite(f"../test/encoder_{i}/{j}.png", im, "png")

    value = encoder.predict(image)

    intermediate_output = get_activations(decoder, value)
    for i, (key, output) in enumerate(intermediate_output.items(), start=1):
        intermediate_images = (output * 255).astype('uint8')
        if len(output.shape) == 4:
            intermediate_images = np.swapaxes(intermediate_images, 0, 3)
            for j in range(len(intermediate_images)):
                im = intermediate_images[j]
                imageio.imwrite(f"../test/decoder_{i}/{j}.png", im, "png")


if __name__ == '__main__':
    main()
