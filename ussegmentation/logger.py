"""Setup logger for the application."""

import logging

LOG_TEMPLATE = (
    "%(asctime)s (%(pathname)s:%(lineno)d) %(levelname)s: %(message)s"
)


def setup_logging(log_file):
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(LOG_TEMPLATE))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_TEMPLATE))
    logger.addHandler(file_handler)
