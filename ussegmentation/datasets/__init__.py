from ussegmentation.datasets.cityscapes import CityscapesDataset
from ussegmentation.datasets.copter import CopterDataset


__ALL__ = ["CityscapesDataset", "CopterDataset"]


def get_dataset_list():
    return [CityscapesDataset, CopterDataset]


def get_dataset_by_name(dataset_name):
    for dataset in get_dataset_list():
        if dataset.name == dataset_name:
            return dataset
