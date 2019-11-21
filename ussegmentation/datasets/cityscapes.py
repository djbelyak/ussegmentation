from torchvision.datasets import Cityscapes


class CityscapesDataset(Cityscapes):
    name = "cityscapes"
    url = "https://165616.selcdn.ru/datasets/cityscapes.zip"
    root = "data\\datasets\\cityscapes"
    class_mapping = {
        -1: 0,
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 2,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0,
        14: 0,
        15: 0,
        16: 0,
        17: 0,
        18: 0,
        19: 0,
        20: 0,
        21: 0,
        22: 0,
        23: 1,
        24: 0,
        25: 0,
        26: 0,
        27: 0,
        28: 0,
        29: 0,
        30: 0,
        31: 0,
        32: 0,
        33: 0,
    }

    def __init__(
        self, transform=None, target_transform=None, transforms=None
    ):
        super(CityscapesDataset, self).__init__(
            CityscapesDataset.root,
            split="train",
            mode="fine",
            target_type="semantic",
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    def __getitem__(self, index):
        image, target = super(CityscapesDataset, self).__getitem__(index)
        target *= 255
        target = target.long()
        for old_class, new_class in CityscapesDataset.class_mapping.items():
            target[target == old_class] = new_class
        return image, target


if __name__ == "__main__":
    from torchvision import transforms

    dataset = CityscapesDataset(
        transform=transforms.Compose([transforms.ToTensor()]),
        target_transform=transforms.Compose([transforms.ToTensor()]),
    )

    import torch

    output_torch = torch.transpose(dataset[0][1], 0, 1)
    output_torch = torch.transpose(output_torch, 1, 2)
    output_numpy = output_torch.detach().numpy()

    from ussegmentation.datasets.utils import remap_classes_to_colors
    import matplotlib.pyplot as plt

    plt.imshow(remap_classes_to_colors(output_numpy))
    plt.show()
