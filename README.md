# emoji-auto-encoder

![Python Version](https://img.shields.io/badge/Python-3.6%2B-blue.svg)
[![Requirements Status](https://requires.io/github/GiantTreeLP/emoji-auto-encoder/requirements.svg?branch=master)](https://requires.io/github/GiantTreeLP/emoji-auto-encoder/requirements/?branch=master)

![Preview](https://github.com/GiantTreeLP/emoji-auto-encoder/raw/master/preview.png)

Compresses Twemoji emojis down to 64 bytes (16 4-bit floating point numbers).

This repository contains an already pretrained model for web use. You can use it as-is without training by hosting
the `www` directory on a web server.

To test the shipped model, just [follow this link](https://gianttreelp.github.io/emoji-auto-encoder/www/).

## Installation

1. Clone this repository.
1. Install Python 3.6 or later and the accompanying `pip3` module.
1. (Optional) Create a `virtualenv` for this project.
1. Run `pip3 install -r requirements.txt`.
1. (Optional) Install the [GPU support libraries](https://www.tensorflow.org/install/gpu) to use your GPU to train the model.
1. (Optional) (If you installed the GPU libraries) Install the `tensorflow-gpu` package using `pip3 install tensorflow-gpu`.

## Preparation

1. Navigate to the `src` directory.
1. Run `python3 svg2png.py` to download and convert the images to a usable format.

## Training

Now we are all set, time to train the network:

1. Navigate to the `src` directory.
1. Run `python3 autoencoder.py`

## Prepare the model for web use

To prepare the trained model for use in the web, use the `tensorflowjs_converter`.

If you have used `virtualenv` to create a virtual environment on Windows, you can find the `tensorflowjs_converter.exe` file in `<virtualenv directory>\Scripts\tensorflowjs_converter.exe`.  
On other operating systems, the binary should already be in your $PATH and ready to be used.

- `<tensorflowjs_converter> --input_format keras --output_format tfjs_layers_model logs\<latest directory>\model.h5 www`

## Use the model in your web browser

Due to the use of [`tfjs`](https://github.com/tensorflow/tfjs), you have to host the `www` directory on a web server.  
Just open the `index.html` file in your browser and use the model or design your own page for it.

## URLs

### Emojis

- https://github.com/twitter/twemoji

### Unicode listing

- https://unicode.org/emoji/charts/full-emoji-list.html
- https://unicode.org/Public/emoji/11.0/emoji-test.txt

### Keras

- https://blog.keras.io/building-autoencoders-in-keras.html
