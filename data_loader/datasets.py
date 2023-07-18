import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset


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
