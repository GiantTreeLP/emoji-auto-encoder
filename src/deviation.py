import glob

import PIL.Image as Image
import numpy as np


def determine_deviation(original: np.ndarray, glob_str: str):
    compressed_images = []
    for file in glob.glob(glob_str):
        with open(file, "rb") as f:
            image = Image.open(f).load()
        compressed_images.append(image)
    compressed_images = np.reshape(np.array(compressed_images), (-1, 128, 128, 4))
    compressed_images = compressed_images.astype('float32') / 255

    deviation = original - compressed_images
    deviation = np.sum(np.abs(deviation), (1, 2)).flatten().astype(str)
    for d in deviation:
        print(d.replace(".", ","))


def main():
    images = []
    for file in glob.glob("../emojis/twemoji/png/*.png"):
        print(file)
        with open(file, "rb") as f:
            image = Image.open(f).load()
        images.append(image)
    images = np.array(images)
    images = np.reshape(images, (-1, 128, 128, 4))
    images = images.astype('float32') / 255

    print("Original")
    print("==========")
    determine_deviation(images, "../emojis/twemoji/png/*.png")
    print("JPG 20%")
    print("==========")
    determine_deviation(images, "../emojis/twemoji/jpg/*.jpg")
    print("JPG 1%")
    print("==========")
    determine_deviation(images, "../emojis/twemoji/jpg_1/*.jpg")
    print("Neuronales Netz")
    print("==========")
    determine_deviation(images, "../emojis/twemoji/test/*.png")


if __name__ == '__main__':
    main()
