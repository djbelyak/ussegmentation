from ussegmentation.datasets.cityscapes import CityscapesDataset
from ussegmentation.datasets.copter import CopterDataset

__ALL__ = ["CityscapesDataset", "CopterDataset"]


def get_dataset_list():
    return [CityscapesDataset, CopterDataset]
