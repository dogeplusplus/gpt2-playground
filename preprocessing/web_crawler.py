import os
import click
import tarfile
import requests
import urllib.parse

from pathlib import Path
from bs4 import BeautifulSoup


@click.command()
@click.argument("url", type=str)
@click.argument("download_dir", type=click.Path())
def crawl_archives(url, download_dir):
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    links = soup.find_all("a")

    for link in links:
        href = link.get("href")
        if href.endswith(".tar.bz2"):
            file_url = urllib.parse.urljoin(url, href)
            print(file_url)
            response = requests.get(file_url)
            tar_file_path = os.path.join(download_dir, Path(href).name)
            with open(tar_file_path, "wb") as file:
                file.write(response.content)

            with tarfile.open(tar_file_path, "r:bz2") as tar_ref:
                tar_ref.extractall(download_dir)
            os.remove(tar_file_path)
