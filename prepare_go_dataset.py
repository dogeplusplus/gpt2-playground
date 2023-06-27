import tiktoken
import pandas as pd
import numpy as np

from typing import List
from sgfmill import sgf
from pathlib import Path
from rich.progress import track


def format_moves(moves: List[str]) -> str:
    moves = str(moves)
    for char in ["(", ")", "'", ",", "[", "]"]:
        moves = moves.replace(char, "")

    return moves


def read_sgf_moves(sgf_file: Path):
    sgf_game = sgf.Sgf_game.from_bytes(sgf_file.read())
    main_game = sgf_game.get_main_sequence()
    root_node = sgf_game.get_root()

    def get_value(key):
        try:
            return root_node.get(key)
        except Exception:
            return None

    parameters = dict(
        event=get_value("EV"),
        round=get_value("RO"),
        black_player=get_value("PB"),
        white_player=get_value("PW"),
        black_rank=get_value("BR"),
        white_rank=get_value("WR"),
        time_limits=get_value("TM"),
        komi=get_value("KM"),
        result=get_value("RE"),
        rules=get_value("RU"),
        overtime=get_value("OT"),
        moves=format_moves([node.get_move() for node in main_game[1:]])
    )

    return parameters


def export_games_to_csv(sgf_file_path: Path, output_file: Path):
    all_sgf_files = list(Path(sgf_file_path).glob("**/*.sgf"))
    rows = []
    for file in track(all_sgf_files, description="Reading SGF files", total=len(all_sgf_files)):
        row = read_sgf_moves(open(file, "rb"))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)


def encode_moves(moves_df: pd.DataFrame, output_file: Path):
    enc = tiktoken.get_encoding("gpt2")
    moves_df.dropna(subset=["moves", "result"], inplace=True)

    train_df = moves_df.sample(frac=0.9)
    val_df = moves_df.drop(train_df.index)

    train_df["moves"] = train_df["moves"]
    val_df["moves"] = val_df["moves"]

    train_ids = enc.encode_ordinary_batch(train_df["moves"].to_list())
    val_ids = enc.encode_ordinary_batch(val_df["moves"].to_list())

    max_seq_length = max(len(x) for x in train_ids + val_ids)
    pad_token_id = enc.encode(" ")[0]
    train_ids = np.array([x + [pad_token_id] * (max_seq_length - len(x)) for x in train_ids], dtype=np.uint16)
    val_ids = np.array([x + [pad_token_id] * (max_seq_length - len(x)) for x in val_ids], dtype=np.uint16)

    np.save(Path(output_file).with_suffix(".train.npy"), train_ids)
    np.save(Path(output_file).with_suffix(".val.npy"), val_ids)


def main():
    sgf_file_path = str(Path.home() / "Downloads/games")
    output_file = Path("games.csv")

    if not output_file.exists():
        export_games_to_csv(sgf_file_path, output_file)

    encoded_file = "go.bin"
    encode_moves(pd.read_csv(output_file), encoded_file)


if __name__ == "__main__":
    main()
