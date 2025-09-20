from pathlib import Path

import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Adadelta
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset.autoencoder_dataset import AutoEncoderDataset, pil_loader_rgba
from model.autoencoder_model import AutoEncoder
from trainer import Trainer

SEED = 7
TRAIN_PATH = Path("./train")
DATASET_REPETITIONS = 8


def main():
    torch.manual_seed(SEED)

    train_dl, eval_dl = load_dataset()

    model = AutoEncoder(128)
    loss_function = torch.nn.MSELoss()

    print(model)
    print("Parameters requiring training:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    training_epochs = 50000

    main_training_loop(train_dl, eval_dl, loss_function, model, training_epochs)


def main_training_loop(train_dl: DataLoader,
                       eval_dl: DataLoader,
                       loss_function: _Loss,
                       model: AutoEncoder,
                       training_epochs: int):
    trainer = Trainer(
        TRAIN_PATH,
        training_epochs,
        model,
        loss_function,
        optimizer_class=Adadelta,
        train_dataset=train_dl,
        train_batch_size=None,
        eval_dataset=eval_dl,
        eval_interval=200,
        save_interval=1000,
    )

    trainer.load_state()
    trainer.train()


def load_dataset() -> tuple[DataLoader, DataLoader]:
    # Load the dataset
    train_dataset = AutoEncoderDataset("./emojis/twemoji/png/",
                                       loader=pil_loader_rgba,
                                       transform=transforms.ToTensor(),
                                       repetitions=DATASET_REPETITIONS)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=min(len(train_dataset), 1024),
                                  shuffle=True,
                                  num_workers=0)

    eval_dataset = AutoEncoderDataset("./emojis/twemoji/png/",
                                      loader=pil_loader_rgba,
                                      transform=transforms.ToTensor())
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=min(len(eval_dataset), 1024),
                                 shuffle=False,
                                 num_workers=0)
    return train_dataloader, eval_dataloader


if __name__ == '__main__':
    main()
