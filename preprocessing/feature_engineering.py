import numpy as np

from ast import literal_eval
from einops import rearrange
from typing import Tuple, List


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
    moves = literal_eval(moves)
    moves = [m for m in moves if m[0] is not None]
    moves_encoded = [ENCODING["SOS"]] + [ENCODING[m] for m in moves] + [ENCODING["EOS"]]
    return moves_encoded
