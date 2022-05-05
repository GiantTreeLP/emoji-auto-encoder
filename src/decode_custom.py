import glob
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import imageio.v3 as imageio
import numpy as np

from autoencoder import get_model

if __name__ == '__main__':
    images = []
    names = []
    for file in glob.glob("../emojis/twemoji/png/*.png"):
        images.append(imageio.imread(file))
        names.append(os.path.splitext(os.path.basename(file))[0])

    images = np.array(images)
    images = np.reshape(images, (-1, 128, 128, 4))
    images = images.astype('float32') / 255
    _, encoder, decoder = get_model(16)

    prediction = encoder.predict(images)
    print("Encoded:")
    print(prediction.dtype)
    for p in prediction:
        print(p)

    prediction = decoder.predict(prediction)
    prediction = (np.clip(prediction, 0, 1) * 255).astype('uint8')
    print("Decoded and saved.")

    os.makedirs("../emojis/twemoji/test", exist_ok=True)
    for i in range(len(prediction)):
        imageio.imwrite(f"../emojis/twemoji/test/{names[i]}.png", prediction[i], format_hint=".png")
