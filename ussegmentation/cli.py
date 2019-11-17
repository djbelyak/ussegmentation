"""Command line interface for the application."""

import click
import logging
import ussegmentation

import ussegmentation.logger as logger

from ussegmentation.download import Downloader
from ussegmentation.inference import InferenceCreator


log = logging.getLogger(__name__)


@click.group()
@click.version_option(version=ussegmentation.__VERSION__)
@click.option(
    "--log",
    type=click.Path(),
    help="Path to log file",
    default="segmentation.log",
)
def cli(log):
    logger.setup_logging(log)


@cli.command()
@click.argument(
    "arg", type=click.Choice(["datasets", "models"], case_sensitive=False)
)
def get(arg):
    """Get needed data."""
    Downloader(arg).download()


@cli.command()
def train():
    """Train a neuron network on specified dataset."""
    log.error("Train command is not implemented yet")


@cli.command()
@click.argument(
    "model",
    type=click.Choice(
        list(InferenceCreator.inferences.keys()), case_sensitive=False
    ),
    default="empty",
)
@click.option(
    "--input-type",
    type=click.Choice(InferenceCreator.input_types, case_sensitive=False),
    help="Type of input information",
    default="video",
)
@click.option(
    "--input-file", type=click.Path(), help="Path to input file", default=""
)
@click.option(
    "--output-file", type=click.Path(), help="Path to output file", default=""
)
@click.option(
    "--show/--no-show", default=True, help="Show preview in a cv window"
)
def inference(model, input_type, input_file, output_file, show):
    """Load pre-trained network and produce a result.

    MODEL is a model of pre-trainged network.
    """
    try:
        inference = InferenceCreator(model).create_inference()
        inference.run(input_type, input_file, output_file, show)
    except Exception as e:
        logging.error(e, exc_info=True)
