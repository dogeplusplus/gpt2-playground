import os
import click
import py7zr

from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--source", type=click.Path(exists=True))
@click.option("--destination", type=click.Path())
def extract_archives(source: Path, destination: Path):
    destination.mkdir(exist_ok=True)
    all_archives = list(source.rglob("**/*7z*"))
    pool = Pool(processes=os.cpu_count())
    result = pool.starmap(extract_archive, zip(all_archives, repeat(destination)))

    return result


def extract_archive(archive: Path, destination: Path):
    try:
        archive_path = Path(archive)

        archive = py7zr.SevenZipFile(archive_path, mode="r")
        archive.extractall(path=destination)
        logger.info(f"Extracted {archive_path} to {destination}")
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")


if __name__ == "__main__":
    extract_archives()
