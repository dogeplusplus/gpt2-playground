import os
import torch
import click
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from data_loader.dataset import GoDataset, huggingface_dataset, collate_fn
from models.gpt import GPTConfig, GPT, BoardEncoder, BoardEncoderConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"


class Go9x9DataModule(pl.LightningDataModule):
    def __init__(self, file: str, batch_size: int, valid_size: float = 0.2):
        super().__init__()
        self.file = file
        self.batch_size = batch_size
        self.dataset = huggingface_dataset(file)
        self.dataset = self.dataset.train_test_split(test_size=valid_size)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 2,
            collate_fn=collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
            collate_fn=collate_fn,
        )
        return val_loader


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, train_file: str, val_file: str, batch_size: int):
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
        decoder_args: dict,
        learning_rate: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        encoder_args: dict = None,
    ):

        super().__init__()
        gpt_config = GPTConfig(**decoder_args)
        encoder = None
        if encoder_args is not None:
            encoder_config = BoardEncoderConfig(**encoder_args)
            encoder = BoardEncoder(encoder_config)

        self.model = GPT(gpt_config, encoder)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        _, loss = self.model(
            batch["input_ids"],
            targets=batch["target_ids"],
            board_states=batch["board_history"],
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self.model(
            batch["input_ids"],
            targets=batch["target_ids"],
            board_states=batch["board_history"],
        )
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
@click.option("--n_layer", type=int, default=12)
@click.option("--n_head", type=int, default=12)
@click.option("--n_embd", type=int, default=768)
@click.option("--bias", type=bool, default=True)
@click.option("--dropout", type=float, default=0.0)
@click.option("--vocab_size", type=int, default=1600)
@click.option("--accumulation_steps", type=int, default=5 * 8)
@click.option("--batch_size", type=int, default=96)
@click.option("--block_size", type=int, default=1024)
@click.option("--epochs", type=int, default=100)
@click.option("--learning_rate", type=float, default=6e-4)
@click.option("--beta1", type=float, default=0.9)
@click.option("--beta2", type=float, default=0.95)
@click.option("--weight_decay", type=float, default=0.1)
@click.option("--encoder_decoder", type=bool, default=True)
@click.option("--encoder_n_layer", type=int, default=2)
@click.option("--encoder_n_head", type=int, default=16)
@click.option("--encoder_n_embd", type=int, default=256)
def main(
    n_layer: int,
    n_head: int,
    n_embd: int,
    bias: bool,
    dropout: float,
    vocab_size: int,
    accumulation_steps: int,
    batch_size: int,
    block_size: int,
    epochs: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    encoder_decoder: bool,
    encoder_n_layer: int,
    encoder_n_head: int,
    encoder_n_embd: int,
):
    decoder_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
        dropout=dropout,
    )

    encoder_args = None
    if encoder_decoder:
        encoder_args = dict(
            n_layers=encoder_n_layer,
            n_head=encoder_n_head,
            n_embd=encoder_n_embd,
        )

    gpt_model = LitGPT(
        decoder_args,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        encoder_args,
    )

    import torch._dynamo.config as dyn
    dyn.suppress_errors = True
    gpt_model = torch.compile(gpt_model)

    go_9x9 = Go9x9DataModule("data/processed/9x9_games.csv", batch_size=batch_size)
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
        datamodule=go_9x9,
    )


if __name__ == "__main__":
    main()
