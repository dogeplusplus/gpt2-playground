import tiktoken
import numpy as np
import pandas as pd

from pathlib import Path
from einops import rearrange
from typing import Tuple, List

from constants import ENCODING, DECODING


def grid_encoding(moves: List[Tuple[str, Tuple[int, int]]]) -> int:
    moves = [m for m in moves if m[0] is not None]
    moves_encoded = [ENCODING["SOS"]] + [ENCODING[m] for m in moves] + [ENCODING["EOS"]]
    return moves_encoded


def encode_moves_fixed(moves_df: pd.DataFrame, output_file: Path, max_seq_length: int = 512) -> List[int]:
    moves_df.dropna(subset=["moves", "result"], inplace=True)

    train_df = moves_df.sample(frac=0.9)
    val_df = moves_df.drop(train_df.index)

    train_ids = [grid_encoding(m) for m in train_df["moves"].to_list()]
    val_ids = [grid_encoding(m) for m in val_df["moves"].to_list()]

    pad_token = 0
    train_ids = np.array(
        [x + [pad_token] * (max_seq_length - len(x))
         for x in train_ids if len(x) <= max_seq_length],
        dtype=np.uint16,
    )
    val_ids = np.array(
        [x + [pad_token] * (max_seq_length - len(x))
         for x in val_ids if len(x) <= max_seq_length],
        dtype=np.uint16,
    )

    np.save(Path(output_file).with_suffix(".train.npy"), train_ids)
    np.save(Path(output_file).with_suffix(".val.npy"), val_ids)


def encode_moves_nlp(moves_df: pd.DataFrame, output_file: Path):
    enc = tiktoken.get_encoding("gpt2")
    moves_df.dropna(subset=["moves", "result"], inplace=True)

    train_df = moves_df.sample(frac=0.9)
    val_df = moves_df.drop(train_df.index)

    train_ids = enc.encode_ordinary_batch(train_df["moves"].to_list())
    val_ids = enc.encode_ordinary_batch(val_df["moves"].to_list())

    max_seq_length = max(len(x) for x in train_ids + val_ids)
    pad_token_id = enc.encode(" ")[0]
    train_ids = np.array([x + [pad_token_id] * (max_seq_length - len(x)) for x in train_ids], dtype=np.uint16)
    val_ids = np.array([x + [pad_token_id] * (max_seq_length - len(x)) for x in val_ids], dtype=np.uint16)

    np.save(Path(output_file).with_suffix(".train.npy"), train_ids)
    np.save(Path(output_file).with_suffix(".val.npy"), val_ids)


def process_result(result: str) -> Tuple[int, int]:
    outcome = {
        "B": 0,
        "W": 1,
        "d": 2,
    }
    point_difference = None
    if result == "Draw":
        player = "d"
    else:
        player, diff = result.split("+")
        if diff not in ["Resign", "Time", "Illegal"]:
            point_difference = float(diff)

    return outcome[player], point_difference


def board_state_history(moves: List[int]) -> np.ndarray:
    moves_decoded = [DECODING[m] for m in moves]
    player_to_idx = {"b": 1, "w": 2}
    board = np.zeros((len(moves), 9, 9), dtype=np.uint8)
    for i, (player, move) in enumerate(moves_decoded):
        if move is None:
            continue
        board[i:, move[0], move[1]] = player_to_idx[player]

    num_classes = 3
    one_hot = np.eye(num_classes, dtype=np.uint8)[board]
    one_hot = rearrange(one_hot, "t h w c -> t c h w")
    return one_hot
