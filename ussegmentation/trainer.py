import logging


class Trainer:
    """Class to perform a training of neuron network."""

    def __init__(self, model, dataset, model_file):
        """Create a Trainer object."""
        self.log = logging.getLogger(__name__)

    def train(self):
        """Perform the training."""
        self.log.info("Start the training")

        self.log.info("Well done")
