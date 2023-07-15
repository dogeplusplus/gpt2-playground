import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import Dataset
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from models.gpt import GPTConfig, GPT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"


class ShakespeareDataset(Dataset):
    def __init__(self, data, block_size):
        super().__init__()
        self.block_size = block_size
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        start = np.random.randint(0, len(self.data) - self.block_size)
        end = start + self.block_size
        x = torch.from_numpy(self.data[start:end].astype(np.int64))
        y = torch.from_numpy(self.data[start+1:end+1].astype(np.int64))

        return x, y


class GoDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.concatenate([self.data[idx, :-1], np.array([0])]).astype(np.int64))
        y = torch.from_numpy(np.concatenate([self.data[idx, 1:], np.array([0])]).astype(np.int64))

        return x, y


class GoCsvDataset(Dataset):
    def __init__(self, csv_path: Path):
        super().__init__()
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, batch_size):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file

        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_data = np.load(self.train_file).astype(np.int16)
        self.val_data = np.load(self.val_file).astype(np.int16)

    def train_dataloader(self):
        train_ds = GoDataset(self.train_data)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 2,
        )
        return train_loader

    def val_dataloader(self):
        val_ds = GoDataset(self.val_data)
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
        )
        return val_loader


class LitGPT(pl.LightningModule):
    def __init__(
        self,
        model_args: dict,
        learning_rate: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
    ):

        super().__init__()
        gpt_config = GPTConfig(**model_args)
        self.model = GPT(gpt_config)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.model(x, targets=y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.model(x, targets=y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            self.weight_decay,
            self.learning_rate,
            (self.beta1, self.beta2),
            self.device,
        )
        return optimizer


def main():
    n_layer = 12
    n_head = 12
    n_embd = 768
    bias = True
    dropout = 0.0
    vocab_size = 1600
    accumulation_steps = 5 * 8
    batch_size = 96
    block_size = 1024

    epochs = 100
    learning_rate = 6e-4
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 0.1

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
        dropout=dropout,
    )

    gpt_model = LitGPT(
        model_args,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
    )
    gpt_model = torch.compile(gpt_model)

    train_file = "data/processed/go.train.npy"
    val_file = "data/processed/go.val.npy"

    shakespeare = GPTDataModule(
        train_file,
        val_file,
        batch_size=batch_size,
    )
    wandb_logger = WandbLogger(project="gopt", log_model=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="gopt",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accumulate_grad_batches=accumulation_steps,
        precision="bf16",
        strategy="ddp",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir="checkpoints",
        logger=wandb_logger,
    )
    trainer.fit(
        model=gpt_model,
        datamodule=shakespeare,
    )


if __name__ == "__main__":
    main()
