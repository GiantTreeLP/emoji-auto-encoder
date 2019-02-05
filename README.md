# emoji-auto-encoder

## Installation

1. Clone this repository.
1. Install Python 3.6 or later and the accompanying `pip3` module.
1. Run `pip3 install -r requirements.txt`.
1. (Optional) Install the [GPU support libraries](https://www.tensorflow.org/install/gpu) to use your GPU to train the model.
1. (Optional) (If you installed the GPU libraries) Install the `tensorflow-gpu` package using `pip3 install tensorflow-gpu`.

## Preparation

1. Navigate to the `src` directory.
1. Run `python3 svg2png.py` to download and convert the images to a usable format.

## Running

Now we are all set, time to train the network:

1. Navigate to the `src` directory.
1. Run `python3 autoencoder.py`

## URLs

### Emojis

- https://github.com/twitter/twemoji

### Unicode listing

- https://unicode.org/emoji/charts/full-emoji-list.html
- https://unicode.org/Public/emoji/11.0/emoji-test.txt

### Keras

- https://blog.keras.io/building-autoencoders-in-keras.html