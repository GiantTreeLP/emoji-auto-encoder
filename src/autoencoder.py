from pathlib import Path

import torch
from torch import Tensor, autocast
from torch.amp import GradScaler
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from dataset.autoencoder_dataset import AutoEncoderDataset, pil_loader_rgba
from model.autoencoder_model import AutoEncoder
from model.train_state import TrainState

SEED = 7
MODEL_PATH = Path("./model.safetensors")
OPTIMIZER_PATH = Path("./optimizer.pth")
SCALER_PATH = Path("./scaler.pth")
DATASET_REPETITIONS = 8


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda")

    images = load_dataset(device)

    loss_function, model, optimizer, train_state = AutoEncoder.load(MODEL_PATH, OPTIMIZER_PATH, device)

    training_epochs = 50000

    main_training_loop(images, loss_function, model, optimizer, train_state, training_epochs)


def main_training_loop(images: list[tuple[int, list[Tensor]]],
                       loss_function: _Loss,
                       model: AutoEncoder,
                       optimizer: Optimizer,
                       train_state: TrainState,
                       training_epochs: int):
    summary_writer = SummaryWriter()

    summary_writer.add_graph(model, images[0][1][0][:1])
    grid = make_grid(images[0][1][0][:4])
    summary_writer.add_image("1_Original", grid, 0)

    scaler = GradScaler()
    if SCALER_PATH.exists():
        scaler.load_state_dict(torch.load(SCALER_PATH))

    epochs = trange(train_state.epoch + 1, train_state.epoch + training_epochs + 1, desc="Epochs",
                    initial=train_state.epoch, )
    for epoch in epochs:
        batches = tqdm(images, desc="Batches", disable=True)
        loss = 0
        for batch, data in batches:
            in_data, out_data = data
            with autocast(device_type="cuda", dtype=torch.float16, enabled=True, cache_enabled=True):
                output = model(in_data)
                loss = loss_function(output, in_data)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epochs.set_postfix({"Loss": loss.item()})

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                input_parameters = images[0][1][0][:4]
                encoded_parameters = model.encoder(input_parameters)
                preview = model.decoder(encoded_parameters)
                preview = preview.cpu().detach()
                grid = make_grid(preview)
                summary_writer.add_image("2_Preview", grid, epoch)
            model.train()

        if epoch % 1000 == 0:
            # Save the model every 1000 epochs
            model.save(train_state=train_state)

            model.eval()
            with torch.no_grad():
                full_preview_input = images[0][1][0]
                input_parameters = full_preview_input[:len(full_preview_input) // DATASET_REPETITIONS]
                encoded_parameters = model.encoder(input_parameters)
                preview = model.decoder(encoded_parameters)
                preview = preview.cpu().detach()
                grid = make_grid(preview)
                summary_writer.add_image("3_Full Preview", grid, epoch)
            model.train()

        summary_writer.add_scalar("Loss", loss.item(), epoch)

        train_state.loss = loss.item()
        train_state.epoch = epoch
        torch.save(optimizer, OPTIMIZER_PATH)
        torch.save(scaler, SCALER_PATH)

    summary_writer.close()

    model.save(train_state=train_state)


def load_dataset(device) -> list[tuple[int, list[Tensor]]]:
    # Load the dataset
    dataset = AutoEncoderDataset("./emojis/twemoji/png/",
                                 loader=pil_loader_rgba,
                                 transform=transforms.ToTensor(),
                                 repetitions=DATASET_REPETITIONS)
    dataloader = DataLoader(dataset,
                            batch_size=min(len(dataset), 1024),
                            shuffle=True,
                            num_workers=0)
    # Extract all the data out of the dataloader
    images = list(iter(dataloader))
    # Move data to device
    for i, d in enumerate(images):
        for j, t in enumerate(d):
            images[i][j] = t.to(device)
    images = list(enumerate(images))
    return images


if __name__ == '__main__':
    main()
