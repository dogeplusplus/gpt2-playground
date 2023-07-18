import os
import torch
import click
import numpy as np
import pytorch_lightning as pl

from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from data_loader.datasets import GoDataset
from models.gpt import GPTConfig, GPT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"


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


@click.command(help="Train a GPT model on Go games")
@click.argument("n_layer", type=int, default=12)
@click.argument("n_head", type=int, default=12)
@click.argument("n_embd", type=int, default=768)
@click.argument("bias", type=bool, default=True)
@click.argument("dropout", type=float, default=0.0)
@click.argument("vocab_size", type=int, default=1600)
@click.argument("accumulation_steps", type=int, default=5 * 8)
@click.argument("batch_size", type=int, default=96)
@click.argument("block_size", type=int, default=1024)
@click.argument("epochs", type=int, default=100)
@click.argument("learning_rate", type=float, default=6e-4)
@click.argument("beta1", type=float, default=0.9)
@click.argument("beta2", type=float, default=0.95)
@click.argument("weight_decay", type=float, default=0.1)
def main(
    n_layer,
    n_head,
    n_embd,
    bias,
    dropout,
    vocab_size,
    accumulation_steps,
    batch_size,
    block_size,
    epochs,
    learning_rate,
    beta1,
    beta2,
    weight_decay,
):
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
