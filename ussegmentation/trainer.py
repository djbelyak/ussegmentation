import logging
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from ussegmentation.models import get_model_by_name
from ussegmentation.datasets import get_dataset_by_name
from torchvision import transforms


class Trainer:
    """Class to perform a training of neuron network."""

    def __init__(self, model, dataset, model_file):
        """Create a Trainer object."""
        self.log = logging.getLogger(__name__)
        self.log.info(
            "Model %s (%s) on %s dataset", model, model_file, dataset
        )

        self.model = get_model_by_name(model)
        self.model_file = Path(model_file)
        if self.model_file.exists():
            self.model.load_state_dict(torch.load(self.model_file))

        self.dataset_name = dataset
        self.prepare_training()

    def prepare_training(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = get_dataset_by_name(self.dataset_name)(
            transform=transform, target_transform=transform
        )
        self.dataset_sizes = {
            "train": int(0.8 * len(self.dataset)),
            "val": len(self.dataset) - int(0.8 * len(self.dataset)),
        }
        train, val = torch.utils.data.random_split(
            self.dataset,
            (self.dataset_sizes["train"], self.dataset_sizes["val"]),
        )
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                train, batch_size=10, shuffle=True, num_workers=4
            ),
            "val": torch.utils.data.DataLoader(
                val, batch_size=10, shuffle=True, num_workers=4
            ),
        }
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def train(self, num_epochs):
        """Perform the training."""
        self.log.info("Start the training")

        since = time.time()

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_loss = float("inf")
        self.best_acc = 0.0

        for epoch in range(num_epochs):
            self.log.info("Epoch %d/%d", (epoch + 1), num_epochs)
            self.train_one_epoch()

        time_elapsed = time.time() - since
        self.log.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        self.log.info("Best val Loss: {:4f}".format(self.best_loss))

        # save best model weights
        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.model.state_dict(), self.model_file)
        self.log.info("Well done")

    def train_one_epoch(self):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                self.model.train()  # Set model to training mode
            else:
                self.model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for images, segments in self.dataloaders[phase]:
                images = images.to(self.device)
                segments = segments.to(self.device).long()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = self.model(images)

                    max_values, max_indices = torch.max(segments, 1)

                    loss = self.criterion(outputs, max_indices)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                _, class_id = torch.max(outputs, 1)
                running_corrects += torch.sum(class_id == segments) / (
                    class_id.size(0) * class_id.size(1) * class_id.size(2)
                )

            if phase == "train":
                self.scheduler.step()

            epoch_loss = running_loss / self.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

            self.log.info(
                "{} Loss: {:.4f} Accuracy: {:.4f}".format(
                    phase, epoch_loss, epoch_acc
                )
            )

            # deep copy the model
            if phase == "val" and epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), self.model_file)

