"""Command line interface for the application."""

import click
import logging
import ussegmentation

import ussegmentation.logger as logger

from ussegmentation.download import Downloader
from ussegmentation.inference import Inference
from ussegmentation.trainer import Trainer
from ussegmentation.datasets import get_dataset_list
from ussegmentation.models import get_model_list, get_model_by_name


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
@click.argument(
    "model",
    type=click.Choice(
        list([model.name for model in get_model_list()]), case_sensitive=False
    ),
    default="empty",
)
@click.option(
    "--dataset",
    type=click.Choice(
        [dataset.name for dataset in get_dataset_list()], case_sensitive=False
    ),
)
@click.option(
    "--model-file",
    type=click.Path(),
    help="Path to result model file",
    default="",
)
def train(model, dataset, model_file):
    """Train a neuron network on specified dataset."""
    try:
        trainer = Trainer(model, dataset, model_file)
        trainer.train()
    except Exception as e:
        logging.error(e, exc_info=True)


@cli.command()
@click.argument(
    "model",
    type=click.Choice(
        list(model.name for model in get_model_list()), case_sensitive=False
    ),
    default="empty",
)
@click.option(
    "--model-file",
    type=click.Path(),
    help="Path to pre-trained model file",
    default="",
)
@click.option(
    "--input-type",
    type=click.Choice(Inference.input_types, case_sensitive=False),
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
def inference(model, model_file, input_type, input_file, output_file, show):
    """Load pre-trained network and produce a result.

    MODEL is a model of pre-trainged network.
    """
    try:
        model = get_model_by_name(model)
        inference = Inference(model, model_file)
        inference.run(input_type, input_file, output_file, show)
    except Exception as e:
        logging.error(e, exc_info=True)
