from collections import OrderedDict

from torch.nn import Module, Sequential, Conv2d, ReLU, Flatten, Dropout, Linear, Tanh, \
    Unflatten, Sigmoid, ConvTranspose2d


class AutoEncoder(Module):
    def __init__(self, vector_len: int = 128, *args, **kwargs):
        super(AutoEncoder, self).__init__(*args, **kwargs)

        self.encoder = Sequential(OrderedDict([
            # Input: 4x128x128
            ("Convolution_64x64", Conv2d(4, 64, 7, stride=2, padding=3)),
            ("Activation_1", ReLU()),
            ("Convolution_32x32", Conv2d(64, 32, 5, stride=2, padding=2)),
            ("Activation_2", ReLU()),
            ("Convolution_16x16", Conv2d(32, 16, 3, stride=2, padding=1)),
            ("Activation_3", ReLU()),
            ("Convolution_8x8", Conv2d(16, 8, 3, stride=2, padding=1)),
            ("Activation_4", ReLU()),
            (f"Flatten_1x512", Flatten()),
            (f"Dense_{vector_len}", Linear(512, vector_len)),
            ("Activation_Last", Tanh()),
        ]))

        self.decoder = Sequential(OrderedDict([
            ("Dropout", Dropout(0.1)),
            ("Dense_1024", Linear(vector_len, 16 * 16 * 4)),
            ("Activation_First", Tanh()),
            ("Unflatten_16x16", Unflatten(1, (4, 16, 16))),
            ("Convolution_Cleanup_16x16", ConvTranspose2d(4, 16, 3, stride=1, padding=1, output_padding=0)),
            ("Activation_1", ReLU()),
            ("Convolution_16x16", ConvTranspose2d(16, 32, 3, stride=1, padding=1, output_padding=0)),
            ("Activation_2", ReLU()),
            ("Convolution_32x32", ConvTranspose2d(32, 64, 5, stride=2, padding=2, output_padding=1)),
            ("Activation_3", ReLU()),
            ("Convolution_64x64", ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1)),
            ("Activation_4", ReLU()),
            ("Convolution_128x128", ConvTranspose2d(64, 4, 7, stride=2, padding=3, output_padding=0)),
            ("Activation_5", ReLU()),
            ("Convolution_Cleanup_128x128", ConvTranspose2d(4, 4, 8, stride=1, padding=3, output_padding=0)),
            ("Activation_Last", Sigmoid()),
        ]))

    def forward(self, x):
        return self.decoder(self.encoder(x))
