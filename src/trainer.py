from os import PathLike
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model
from torch import GradScaler
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import trange, tqdm

from model.train_state import TrainState

MODEL_FILE = "model.safetensors"
OPTIMIZER_FILE = "optimizer.pth"
STATE_FILE = "state.json"
SCALER_FILE = "scaler.pth"


class Trainer:
    def __init__(self,
                 state_directory: PathLike,
                 epochs: int,
                 model: Module,
                 loss_function: _Loss,
                 optimizer_class: type[Optimizer],
                 train_dataset: DataLoader,
                 train_batch_size: int = None,
                 eval_dataset: DataLoader = None,
                 eval_interval: int = 100,
                 save_interval: int = 1000,
                 ):
        self.state_directory = Path(state_directory)

        self.epochs = epochs
        self.model = model
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.train_dataset = train_dataset
        # Extract a single batch for evaluation
        self.eval_dataset = next(iter(eval_dataset))[0] if eval_dataset is not None else torch.stack(
            [next(iter(train_dataset))[1][0][0]])

        self.optimizer = self.optimizer_class(params=self.model.parameters())
        self.scaler = GradScaler()
        self.state = TrainState()

        self.train_batch_size = train_batch_size if train_batch_size is not None else len(train_dataset)
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        self.summary_writer = SummaryWriter()

    def load_state(self):
        if not self.state_directory.exists():
            self.state_directory.mkdir(parents=True, exist_ok=True)
            return

        if not self.state_directory.is_dir():
            raise ValueError(f"State directory {self.state_directory} is not a directory!")

        # Load model
        model_path = self.state_directory / MODEL_FILE
        if model_path.exists():
            load_model(self.model, model_path)

        # Load training state
        state_path = self.state_directory / STATE_FILE
        if state_path.exists():
            self.state = TrainState.load(state_path)

        # Load optimizer state
        optimizer_path = self.state_directory / OPTIMIZER_FILE
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))

        # Load scaler state
        scaler_path = self.state_directory / SCALER_FILE
        if scaler_path.exists():
            self.scaler.load_state_dict(torch.load(scaler_path))

    def save_state(self):
        if not self.state_directory.is_dir():
            self.state_directory.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = self.state_directory / MODEL_FILE
        save_model(self.model, str(model_path))

        # Save training state
        state_path = self.state_directory / STATE_FILE
        self.state.save(state_path)

        # Save optimizer state
        optimizer_path = self.state_directory / OPTIMIZER_FILE
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Save scaler state
        scaler_path = self.state_directory / SCALER_FILE
        torch.save(self.scaler.state_dict(), scaler_path)

    def _select_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        self.model.to(device)
        self.loss_function.to(device)
        self.scaler = GradScaler(device.type)
        self._prepare_dataset(device)

    def _prepare_dataset(self, device: torch.device):
        self.train_dataset = list(enumerate(list(map(lambda x: x.to(device), batch)) for batch in self.train_dataset))
        self.eval_dataset = self.eval_dataset.to(device)

    def _before_training(self):
        self._select_device()
        self.model.train()
        self.summary_writer.add_graph(self.model, torch.reshape(self.eval_dataset[0],
                                                                (-1, *self.eval_dataset[0].shape)))
        grid = make_grid(self.eval_dataset[:4])
        self.summary_writer.add_image("1_Original", grid, 0)
        del grid

    def _after_training(self):
        self.summary_writer.flush()
        self.summary_writer.close()

    def _after_epoch(self, epoch: int, loss: float):
        self.summary_writer.add_scalar("Train/Loss", loss, epoch)

        self.state.epoch = epoch
        self.state.loss = loss

        if epoch % self.eval_interval == 0:
            self.model.eval()

            with torch.no_grad():
                preview = self.model(self.eval_dataset[:4]).cpu().detach()
                grid = make_grid(preview)
                self.summary_writer.add_image("2_Preview", grid, epoch)
                del preview
                del grid

            self.model.train()

            self.summary_writer.flush()

        if epoch % self.save_interval == 0:
            self.save_state()

            self.model.eval()
            with torch.no_grad():
                preview = self.model(self.eval_dataset).cpu().detach()
                grid = make_grid(preview)
                self.summary_writer.add_image("3_Full Preview", grid, epoch)
                del preview
                del grid

            self.model.train()

    def train(self):
        self._before_training()

        self._train_loop()

        self._after_training()
        self.save_state()

    def _train_loop(self):
        epochs = trange(self.state.epoch + 1, self.state.epoch + self.epochs + 1, desc="Epochs",
                        initial=self.state.epoch)
        # Go through each epoch
        for epoch in epochs:
            # Go through each batch
            batches = tqdm(self.train_dataset, desc="Batches", disable=True)
            loss = 0
            for batch, data in batches:
                in_data, out_data = data
                self.optimizer.zero_grad()
                with torch.autocast(device_type=in_data.device.type, dtype=torch.float16, enabled=True,
                                    cache_enabled=True):
                    output = self.model(in_data)
                    loss = self.loss_function(output, in_data)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            loss = loss.item()
            self._after_epoch(epoch, loss)
            epochs.set_postfix({"Loss": loss})
