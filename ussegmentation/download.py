"""Download all needed things."""
import logging
import requests
import zipfile

from pathlib import Path

from ussegmentation.datasets import get_dataset_list


class Downloader:
    """Class to download all needed data."""

    def __init__(self, arg):
        self.log = logging.getLogger(__name__)
        self.arg = arg
        self.datasets = get_dataset_list()
        self.models_url = "https://165616.selcdn.ru/datasets/models.zip"

    def download(self):
        """Main download function."""
        if self.arg == "datasets":
            self.download_datasets()
        if self.arg == "models":
            self.download_models()

    def download_datasets(self):
        """Download all datasets."""
        self.log.info("Downloading datasets")
        # create datasets dir if not exists
        main_dir = Path("data", "datasets")
        if not main_dir.exists():
            main_dir.mkdir(parents=True)

        # for each dataset
        for dataset in self.datasets:
            dataset_dir = main_dir.joinpath(dataset.name)
            if dataset_dir.exists():
                continue

            self.log.info("Downloading '%s' dataset", dataset.name)
            dataset_dir.mkdir()

            dataset_archive = dataset_dir.joinpath(dataset.url.split("/")[-1])

            with requests.get(dataset.url, stream=True) as r:
                r.raise_for_status()
                with open(dataset_archive, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            with zipfile.ZipFile(dataset_archive, "r") as zip_ref:
                zip_ref.extractall(dataset_dir)

            dataset_archive.unlink()

        self.log.info("Done!")

    def download_models(self):
        """Download all models."""
        self.log.info("Downloading models")
        # create datasets dir if not exists
        main_dir = Path("data", "models")
        if not main_dir.exists():
            main_dir.mkdir(parents=True)

        models_archive = main_dir.joinpath(self.models_url.split("/")[-1])

        with requests.get(self.models_url, stream=True) as r:
            r.raise_for_status()
            with open(models_archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        with zipfile.ZipFile(models_archive, "r") as zip_ref:
            zip_ref.extractall(main_dir)

        models_archive.unlink()

        self.log.info("Done!")
