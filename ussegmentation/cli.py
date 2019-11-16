"""Command line interface for the application."""

import click
import logging

import ussegmentation

import ussegmentation.logger as logger


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
def get():
    """Get needed data."""
    log.error("Get command is not implemented yet")


@cli.command()
def train():
    """Train a neuron network on specified dataset."""
    log.error("Train command is not implemented yet")


@cli.command()
def inference():
    """Load pre-trained network and produce a result."""
    log.error("Inference command is not implemented yet")
