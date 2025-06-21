import os
from pathlib import Path
from typing import Callable, Optional
from torch.utils.data import Dataset
from ds.FilteredDataset import FilteredDataset
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
                # print(f"Full path: {full_path}")
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
                # print(f"Full path: {full_path}")
                self.samples.append((str(full_path), target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target

class SixClassDataset(Dataset):
    """
    Multiclass classification (6 classes):
        0 -> live (0_0)
        1 -> 2D Attack (1_0)
        2 -> 3D Attack (1_1)
        3 -> Digital Manipulation (2_0)
        4 -> Digital Adversarial (2_1)
        5 -> Digital Generation (2_2)
    """
    def __init__(self, protocol_file: str, root_dir: Union[str, Path], transform: Optional[Callable] = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        prefix_map = {
            "0_0": 0,
            "1_0": 1,
            "1_1": 2,
            "2_0": 3,
            "2_1": 4,
            "2_2": 5
        }

        with open(protocol_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                prefix = "_".join(label.split("_")[:2])
                if prefix not in prefix_map:
                    raise ValueError(f"Etiqueta «{label}» no reconocida.")
                target = prefix_map[prefix]
                full_path = self.root_dir / Path(path).name
                self.samples.append((str(full_path), target))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target

class FifteenClassDataset(Dataset):
    """
    Multiclass classification (15 classes):
        0 -> live (0_0_0)
        1–14 -> attack types from 1_0_0 to 2_2_2
    """
    def __init__(self, protocol_file: str, root_dir: Union[str, Path], transform: Optional[Callable] = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        prefix_map = {
            "0_0_0": 0,
            "1_0_0": 1,
            "1_0_1": 2,
            "1_0_2": 3,
            "1_1_0": 4,
            "1_1_1": 5,
            "1_1_2": 6,
            "2_0_0": 7,
            "2_0_1": 8,
            "2_0_2": 9,
            "2_1_0": 10,
            "2_1_1": 11,
            "2_2_0": 12,
            "2_2_1": 13,
            "2_2_2": 14,
        }

        with open(protocol_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                if label not in prefix_map:
                    raise ValueError(f"Etiqueta «{label}» no reconocida.")
                target = prefix_map[label]
                full_path = self.root_dir / Path(path).name
                self.samples.append((str(full_path), target))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
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
    def create(num_classes: int
               , protocol_file: str
               , root_dir: str
               , transform: Optional[Callable] = None
               , is3d: bool = False,
                apply_filters: bool = True) -> Dataset:
        if is3d:            
            return FilteredDataset(
                protocol_file=protocol_file,
                root_dir=root_dir,
                apply_filters=apply_filters,
                transform=transform
            )
        elif num_classes == 2:
            return TwoClassDataset(protocol_file, root_dir, transform)
        elif num_classes == 3:
            return ThreeClassDataset(protocol_file, root_dir, transform)
        elif num_classes == 6:
            return SixClassDataset(protocol_file, root_dir, transform)
        elif num_classes == 15:
            return FifteenClassDataset(protocol_file, root_dir, transform)
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
