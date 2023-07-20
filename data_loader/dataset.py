import torch
import numpy as np
import pandas as pd

from pathlib import Path
from ast import literal_eval
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

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


def process_game(example):
    moves = literal_eval(example["moves"])
    example["moves"] = grid_encoding(moves)
    example["result"], example["point_difference"] = process_result(example["result"])

    return example


def add_board_state_history(example):
    example["board_history"] = board_state_history(example["moves"])
    return example


def huggingface_dataset(path: str) -> Dataset:
    dataset = HFDataset.from_csv(path)
    dataset = dataset.map(process_game)
    dataset = dataset.filter(lambda x: len(x["moves"]) > 0)
    dataset = dataset.map(add_board_state_history)

    return dataset
