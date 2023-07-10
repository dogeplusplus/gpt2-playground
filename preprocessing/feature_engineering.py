import numpy as np

from einops import rearrange
from typing import Tuple, List


def board_state(moves: List[Tuple[str, Tuple[int, int]]]) -> np.ndarray:
    board = np.zeros((19, 19), dtype=np.uint8)
    player_to_idx = {"b": 1, "w": 2}
    for player, (y, x) in moves:
        board[y][x] = player_to_idx[player]

    num_classes = 3
    one_hot = np.eye(num_classes, dtype=np.uint8)[board]
    one_hot = rearrange(one_hot, "h w c -> c h w")
    return one_hot
