import string
import torch

from typing import List, Tuple
from pathlib import Path

from train import LitGPT
from prepare_go_dataset import DECODING


def decode_game(game: List[int]) -> str:
    moves = [DECODING[i] for i in game]
    return moves


def render_board(moves: List[Tuple[str, Tuple[int, int]]]) -> str:
    board = [["·" for _ in range(19)] for _ in range(19)]

    for player, (y, x) in moves:
        board[y][x] = player

    numbers = list(range(1, 20))
    letters = list(string.ascii_uppercase[:20])
    # I is not used in Go
    letters.remove("I")

    print("   " + " ".join(letters))
    for n in numbers:
        print(f"{n:02d} " + " ".join(board[n-1]))


def generate_game(
    model_path: Path,
    temperature: float = 0.8,
    top_k: int = 10,
    max_new_tokens: int = 512,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit_gpt = LitGPT.load_from_checkpoint(model_path, map_location=device)
    gpt = lit_gpt.model
    x = torch.tensor([0], dtype=torch.long, device=device)[None, ...]
    generations = gpt.generate(x, max_new_tokens, temperature, top_k).tolist()
    return decode_game(generations[0])


def main():
    MODEL_PATH = Path.cwd() / "checkpoints/lightning_logs/version_0/checkpoints/gopt.ckpt"
    game = generate_game(MODEL_PATH)
    print(game)


if __name__ == "__main__":
    main()
