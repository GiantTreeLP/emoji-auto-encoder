from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_model, load_model
from torch.nn import Module, Sequential, Conv2d, ReLU, Flatten, Dropout, Linear, Tanh, \
    Unflatten, Sigmoid, ConvTranspose2d
from torch.nn.modules.loss import _Loss, MSELoss

from model.train_state import TrainState


class AutoEncoder(Module):
    def __init__(self, path: Path, vector_len: int = 128, *args, **kwargs):
        super(AutoEncoder, self).__init__(*args, **kwargs)

        self.path = path

        self.encoder = Sequential(OrderedDict([
            # Input: 4x128x128
            ("Convolution_64x64", Conv2d(4, 64, 5, stride=2, padding=2, groups=4)),
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
            ("Dense_64", Linear(vector_len, 8 * 8 * 1)),
            ("Activation_First", Tanh()),
            ("Unflatten_8x8", Unflatten(1, (1, 8, 8))),
            ("Convolution_8x8", ConvTranspose2d(1, 16, 3, stride=1, padding=1, output_padding=0)),
            ("Activation_1", ReLU()),
            ("Convolution_16x16", ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1)),
            ("Activation_2", ReLU()),
            ("Convolution_32x32", ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1)),
            ("Activation_2", ReLU()),
            ("Convolution_64x64", ConvTranspose2d(32, 64, 5, stride=2, padding=2, output_padding=1)),
            ("Activation_3", ReLU()),
            ("Convolution_128x128", ConvTranspose2d(64, 4, 5, stride=2, padding=2, output_padding=1)),
            ("Activation_4", ReLU()),
            ("Convolution_Cleanup_128x128", ConvTranspose2d(4, 4, 3, stride=1, padding=1, output_padding=0)),
            ("Activation_Last", Sigmoid()),
        ]))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def save(self, path: Path = None, train_state: TrainState = None):
        if path is None:
            path = self.path
        save_model(self, path)

        if train_state is not None:
            state_path = path.with_suffix(".state")
            train_state.save(state_path)

    @staticmethod
    def load(model_path: Path, device: torch.device) -> tuple[
        _Loss, "AutoEncoder", TrainState]:
        model = AutoEncoder(model_path)
        if model_path.exists():
            missing, unexpected = load_model(model, model_path)
            if missing or unexpected:
                print(f"Warning: Missing keys: {missing}, Unexpected keys: {unexpected}")
                print("Creating new model")
                model = AutoEncoder(model_path)
        model = model.to(device)
        loss_function = MSELoss().to(device)

        state_path = model_path.with_suffix(".state")
        if state_path.exists():
            train_state = TrainState.load(state_path)
            print(f"Resuming training from epoch {train_state.epoch} with loss {train_state.loss:.4f}")
        else:
            train_state = TrainState()
            print(f"Starting training from epoch {train_state.epoch} with loss {train_state.loss:.4f}")
        print(model)
        print("Parameters requiring training:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        return loss_function, model, train_state
