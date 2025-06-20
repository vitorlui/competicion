import os
from pathlib import Path
from typing import Callable, Optional
from torch.utils.data import Dataset
from PIL import Image

# ========================
#        DATASETS
# ========================
from typing import Union

class TwoClassDataset(Dataset):
    """
    Binary classification:
        0 -> bonafide (0_*)
        1 -> fake (1_* or 2_*)
    """
    # def __init__(self, protocol_file: str, root_dir: str | Path, transform: Optional[Callable] = None) -> None:
    def __init__(self, protocol_file: str, root_dir: Union[str, Path], transform: Optional[Callable] = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        with open(protocol_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                prefix = label.split("_", 1)[0] + "_"
                target = 0 if prefix == "0_" else 1
                full_path = self.root_dir / Path(path).name
                self.samples.append((str(full_path), target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


class ThreeClassDataset(Dataset):
    """
    Multiclass classification:
        0 -> live (0_0_0)
        1 -> physical attack (1_*)
        2 -> digital  attack (2_*)
    """
    def __init__(self, protocol_file: str, root_dir: Union[str, Path], transform: Optional[Callable] = None) -> None:

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        prefixes = {"0_": 0, "1_": 1, "2_": 2}

        with open(protocol_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                prefix = label.split("_", 1)[0] + "_"
                if prefix not in prefixes:
                    raise ValueError(f"Etiqueta «{label}» no reconocida.")
                target = prefixes[prefix]
                full_path = self.root_dir / Path(path).name
                self.samples.append((str(full_path), target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target

# ========================
#        FACTORY
# ========================

class DatasetFactory:
    @staticmethod
    def create(num_classes: int, protocol_file: str, root_dir: str, transform: Optional[Callable] = None) -> Dataset:
        if num_classes == 2:
            return TwoClassDataset(protocol_file, root_dir, transform)
        elif num_classes == 3:
            return ThreeClassDataset(protocol_file, root_dir, transform)
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")

# ========================
#        USAGE
# ========================

# from ds import DatasetFactory  # Ajusta según el nombre del archivo

# dataset = DatasetFactory.create(
#     num_classes=2,
#     protocol_file="phase2/Protocol-val.txt",
#     root_dir="Data-val",
#     transform=tfms
# )
