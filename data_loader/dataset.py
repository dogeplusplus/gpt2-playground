import os
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from ast import literal_eval
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from constants import ENCODING9X9
from data_loader.feature_engineering import board_state_history, process_result, grid_encoding


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


def process_game(example: dict) -> dict:
    moves = literal_eval(example["moves"])
    moves_target = moves[1:] + [" "]
    example["input_ids"] = grid_encoding(moves)
    example["target_ids"] = grid_encoding(moves_target)
    example["result"], example["point_difference"] = process_result(example["result"])

    return example


def add_board_state_history(example: dict) -> dict:
    example["board_history"] = board_state_history(example["input_ids"])
    return example


def huggingface_dataset(path: str, max_game_length: int = 1024) -> Dataset:
    workers = os.cpu_count()
    dataset = HFDataset.from_csv(path)
    dataset = dataset.map(process_game, num_proc=workers)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0, num_proc=workers)
    # shouldn't be any games that are this lone
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_game_length, num_proc=workers)
    dataset = dataset.map(add_board_state_history, num_proc=workers)
    dataset.set_format(type="torch", columns=["input_ids", "target_ids", "board_history", "result"])

    return dataset


def collate_fn(batch: list, encoding: dict = ENCODING9X9):
    padded_encodings = pad_sequence(
        [x["input_ids"] for x in batch],
        batch_first=True,
        padding_value=encoding[" "],
    )
    padded_target_encodings = pad_sequence(
        [x["target_ids"] for x in batch],
        batch_first=True,
        padding_value=encoding[" "],
    )

    t_max = padded_encodings.shape[1]
    padded_history = np.array([
        np.pad(b["board_history"],
               ((0, t_max - b["board_history"].shape[0]), (0, 0), (0, 0), (0, 0)),
               mode="constant", constant_values=-1)
        for b in batch
    ])
    padded_history = torch.tensor(padded_history, dtype=torch.float32)

    batch = {
        "input_ids": padded_encodings,
        "target_ids": padded_target_encodings,
        "board_history": padded_history,
        "result": torch.tensor([x["result"] for x in batch], dtype=torch.float32),
    }

    return batch
