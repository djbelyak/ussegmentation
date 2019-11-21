import torch
import cv2
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

from ussegmentation.datasets.utils import (
    remap_color_to_classes,
    remap_classes_to_colors,
)


class CopterDataset(Dataset):
    name = "copter"
    url = "https://165616.selcdn.ru/datasets/copter.zip"
    root = "data\\datasets\\copter"

    class_mapping = {
        (0, 255, 0): 0,  # other
        (0, 0, 255): 1,  # sky
        (128, 128, 128): 2,  # road
    }

    def replace_color(image, src_color, dest_color):
        src_r, src_g, src_b = src_color
        dest_r, dest_g, dest_b = dest_color
        mask = (
            (image[:, :, 0] == src_r)
            & (image[:, :, 1] == src_g)
            & (image[:, :, 2] == src_b)
        )
        image[:, :, :3][mask] = [dest_r, dest_g, dest_b]
        return image

    def __init__(self, transform=None, target_transform=None):
        self.dir = Path(self.root)
        self.transform = transform
        self.target_transform = target_transform
        self.files = self.get_files()

    def get_files(self):
        files = []
        for filepath in self.dir.iterdir():
            if filepath.name[:5] == "mask_" and filepath.suffix == ".png":
                imagepath = filepath.parent.joinpath(filepath.name[5:])
                if imagepath.exists():
                    files.append(
                        {"image": str(imagepath), "segments": str(filepath)}
                    )

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = cv2.imread(self.files[idx]["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segments = cv2.imread(self.files[idx]["segments"])
        segments = cv2.cvtColor(segments, cv2.COLOR_BGR2RGB)

        remaped_segments = np.zeros(
            (segments.shape[0], segments.shape[1], 1), dtype=np.long
        )
        for color, class_id in self.class_mapping.items():
            remaped_segments = remap_color_to_classes(
                segments, color, remaped_segments, class_id
            )

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            remaped_segments = self.target_transform(remaped_segments)

        return image, remaped_segments


if __name__ == "__main__":
    from torchvision import transforms

    dataset = CopterDataset(
        transform=transforms.Compose([transforms.ToTensor()]),
        target_transform=transforms.Compose([transforms.ToTensor()]),
    )

    import matplotlib.pyplot as plt

    output_torch = torch.transpose(dataset[0][1], 0, 1)
    output_torch = torch.transpose(output_torch, 1, 2)

    output_numpy = output_torch.detach().numpy()

    plt.imshow(remap_classes_to_colors(output_numpy))
    plt.show()
