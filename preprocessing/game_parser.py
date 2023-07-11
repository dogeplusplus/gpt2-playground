import click
import logging
import pandas as pd

from sgfmill import sgf
from pathlib import Path
from rich.progress import track


logger = logging.getLogger(__name__)


def read_sgf_moves(sgf_file: Path):
    sgf_game = sgf.Sgf_game.from_bytes(open(sgf_file, "rb").read())
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
        game_rank=sgf_file.parent.name,
    )

    return parameters


@click.command()
@click.option("--sgf-file-path", type=click.Path(exists=True))
@click.option("--output-file", type=click.Path())
def export_games_to_file(sgf_file_path: Path, output_file: Path):
    all_sgf_files = list(Path(sgf_file_path).glob("**/*.sgf"))
    rows = []
    for file in track(all_sgf_files, description="Reading SGF files", total=len(all_sgf_files)):
        try:
            row = read_sgf_moves(file)
            rows.append(row)
        except ValueError as e:
            logger.warning(f"Could not load file {file} because of {e}")

    df = pd.DataFrame(rows)
    df["moves"] = df["moves"].apply(str)
    df.to_parquet(output_file, index=False)


if __name__ == "__main__":
    export_games_to_file()
