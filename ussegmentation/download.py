"""Download all needed things."""
import logging


class Downloader:
    """Class to download all needed data."""

    def __init__(self, arg):
        self.log = logging.getLogger(__name__)
        self.arg = arg

    def download(self):
        """Main download function."""
        if self.arg == "datasets":
            self.download_datasets()
        if self.arg == "models":
            self.download_models()

    def download_datasets(self):
        """Download all datasets."""
        self.log.info("Downloading datasets")

    def download_models(self):
        """Download all models."""
        self.log.info("Downloading models")
