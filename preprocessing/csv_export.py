import tiktoken
import pandas as pd
import numpy as np
import click
import logging

from ast import literal_eval
from typing import List, Tuple
from sgfmill import sgf
from pathlib import Path
from rich.progress import track

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

logger = logging.getLogger(__name__)


def grid_encoding(moves: List[Tuple[str, Tuple[int, int]]]) -> int:
    moves = literal_eval(moves)
    moves = [m for m in moves if m[0] is not None]
    moves_encoded = [ENCODING[m] for m in moves]
    return moves_encoded


def encode_moves_fixed(moves_df: pd.DataFrame, output_file: Path, max_seq_length: int = 512) -> List[int]:
    moves_df.dropna(subset=["moves", "result"], inplace=True)

    train_df = moves_df.sample(frac=0.9)
    val_df = moves_df.drop(train_df.index)

    train_df["moves"] = train_df["moves"]
    val_df["moves"] = val_df["moves"]

    train_ids = [grid_encoding(m) for m in train_df["moves"].to_list()]
    val_ids = [grid_encoding(m) for m in val_df["moves"].to_list()]

    start_token = ENCODING["SOS"]
    end_token = ENCODING["EOS"]

    # 0 for padding
    train_ids = np.array(
        [[start_token] + x + [end_token] + [0] * (max_seq_length - len(x) - 2)
         for x in train_ids if len(x) <= max_seq_length],
        dtype=np.uint16,
    )
    val_ids = np.array(
        [[start_token] + x + [end_token] + [0] * (max_seq_length - len(x) - 2)
         for x in val_ids if len(x) <= max_seq_length],
        dtype=np.uint16,
    )

    np.save(Path(output_file).with_suffix(".train.npy"), train_ids)
    np.save(Path(output_file).with_suffix(".val.npy"), val_ids)


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
        moves=[node.get_move() for node in main_game],
    )

    return parameters


@click.command()
@click.option("--sgf-file-path", type=click.Path(exists=True))
@click.option("--output-file", type=click.Path())
def export_games_to_csv(sgf_file_path: Path, output_file: Path):
    all_sgf_files = list(Path(sgf_file_path).glob("**/*.sgf"))
    rows = []
    for file in track(all_sgf_files, description="Reading SGF files", total=len(all_sgf_files)):
        try:
            row = read_sgf_moves(open(file, "rb"))
            rows.append(row)
        except ValueError as e:
            logger.warning(f"Could not load file {file} because of {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)


def encode_moves_nlp(moves_df: pd.DataFrame, output_file: Path):
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


if __name__ == "__main__":
    export_games_to_csv()
