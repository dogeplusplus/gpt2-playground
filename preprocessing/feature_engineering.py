import re
import tiktoken
import numpy as np
import pandas as pd

from pathlib import Path
from ast import literal_eval
from einops import rearrange
from typing import Tuple, List, Dict, Any


# For some reason there are moves on the 19th position
ALL_MOVES = [" "] + [
    (p, (i, j)) for p in ('b', 'w') for i in range(25) for j in range(25)
] + [('w', None), ('b', None), "SOS", "EOS"]

ENCODING = {
    move: i for i, move in enumerate(ALL_MOVES)
}

DECODING = {
    i: move for i, move in enumerate(ALL_MOVES)
}


def board_state(moves: List[Tuple[str, Tuple[int, int]]]) -> np.ndarray:
    board = np.zeros((19, 19), dtype=np.uint8)
    player_to_idx = {"b": 1, "w": 2}
    for player, (y, x) in moves:
        board[y][x] = player_to_idx[player]

    num_classes = 3
    one_hot = np.eye(num_classes, dtype=np.uint8)[board]
    one_hot = rearrange(one_hot, "h w c -> c h w")
    return one_hot


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
    # -1 token for padding
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


def extract_features(example: Dict[str, Any]):
    # Skip first move as always none
    example["moves"] = literal_eval(example["moves"])[1:]
    example["board_state"] = np.stack([
        board_state(example["moves"][:i])
        for i in range(1, len(example["moves"])+1)
    ])
    example["moves"] = grid_encoding(example["moves"])

    return example


def process_result(game: Dict[str, Any]):
    outcome = {
        "B": 0,
        "W": 1,
        "d": 2,
    }
    pattern = r"([+-]?\d+(\.\d+)?)"
    result = re.findall(pattern, game["result"])
    game["result"] = outcome[game["result"][0]]
    if result:
        result = result[0][0]
        game["point_difference"] = float(result)
    else:
        game["point_difference"] = None
    return game
