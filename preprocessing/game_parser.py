import click
import logging
import pandas as pd

from sgfmill import sgf
from pathlib import Path
from rich.progress import track


logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


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


def rank_export(directory: Path, output_file: Path):
    all_sgf_files = list(Path(directory).glob("**/*.sgf"))
    rows = []
    for fp in track(all_sgf_files, description="Reading SGF files", total=len(all_sgf_files)):
        try:
            row = read_sgf_moves(fp)
            rows.append(row)
        except ValueError as e:
            logger.warning(f"Could not load file {fp} because of {e}")

    df = pd.DataFrame(rows)
    df["moves"] = df["moves"].apply(str)
    df.to_parquet(output_file, index=False)


@cli.command()
@click.option("--directory", type=click.Path(exists=True))
@click.option("--output-file", type=click.Path())
def single_rank_export(directory: Path, output_file: Path):
    rank_export(directory, output_file)


@cli.command()
@click.option("--base-dir", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
def batch_rank_export(base_dir: Path, output_dir: Path):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for folder in Path(base_dir).iterdir():
        if folder.is_dir():
            output_file = f"{output_dir}/{folder.name}.parquet"
            rank_export(str(folder), output_file)


if __name__ == "__main__":
    cli()
