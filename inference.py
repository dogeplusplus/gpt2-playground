import wandb
import string
import torch
import torch.nn as nn

from typing import List, Tuple
from pathlib import Path

from train import LitGPT
from preprocessing.feature_engineering import DECODING, ENCODING


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
    model: nn.Module,
    temperature: float = 0.8,
    top_k: int = 10,
    max_new_tokens: int = 512,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor([ENCODING["SOS"]], dtype=torch.long, device=device)[None, ...]
    generations = model.generate(x, max_new_tokens, temperature, top_k).tolist()
    return decode_game(generations[0])


def main():
    artifact_uri = "dogeplusplus/gopt/model-k0tgjw3w:v0"
    run = wandb.init(job_type="inference")
    artifact = run.use_artifact(artifact_uri, type="model")
    artifact_dir = artifact.download()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt = LitGPT.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", map_location=device)
    model = gpt.model
    model.eval()

    game = generate_game(model, top_k=1000)
    game = [g for g in game if g != " "]
    print(game)
    render_board(game[1:-1])


if __name__ == "__main__":
    main()
