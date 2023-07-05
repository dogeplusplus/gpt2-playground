import torch

from typing import List
from pathlib import Path

from train import LitGPT
from prepare_go_dataset import DECODING


def decode_game(game: List[int]) -> str:
    moves = [DECODING[i] for i in game]
    return moves


def generate_game(model_path: Path, temperature: float = 0.8, top_k: int = 10, max_new_tokens: int = 512):
    lit_gpt = LitGPT.load_from_checkpoint(model_path)
    gpt = lit_gpt.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor([0], dtype=torch.long, device=device)[None, ...]
    generations = gpt.generate(x, max_new_tokens, temperature, top_k).tolist()
    return decode_game(generations[0])


def main():
    MODEL_PATH = Path.cwd() / "checkpoints/lightning_logs/version_0/checkpoints/gopt.ckpt"
    game = generate_game(MODEL_PATH)
    print(game)


if __name__ == "__main__":
    main()
