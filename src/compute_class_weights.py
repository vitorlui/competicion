import argparse
from time import sleep
from train.Train import train_model
from ds.TransformFactory import TransformFactory
from models.ModelFactory import ModelFactory
from ds.DatasetFactory import DatasetFactory
from torch.utils.data import DataLoader
from ds.DatasetUtils import get_or_compute_class_weights
import argparse
from time import sleep
from train.Train import train_model
from ds.TransformFactory import TransformFactory
from models.ModelFactory import ModelFactory
from ds.DatasetFactory import DatasetFactory
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import ConcatDataset, Subset
from copy import deepcopy
from collections import Counter
import torch

from pathlib import Path

def load_augmented_bonafides(dir_path, label=0, transform=None):
    from torch.utils.data import Dataset
    from PIL import Image

    class AugmentedBonafideDataset(Dataset):
        def __init__(self, root):
            self.root = Path(root)
            self.samples = [
                (str(p), label)
                for p in self.root.glob("*.png") if "_aug" in p.name
            ]
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    return AugmentedBonafideDataset(dir_path)



def print_class_distribution(dataset, name="Dataset"):
    """
    Imprime la distribución de clases de un Dataset o ConcatDataset.

    Args:
        dataset: Un Dataset con muestras del tipo (image, label).
        name (str): Nombre para identificar el dataset en el print.
    """
    labels = []
    
    # Soporte para ConcatDataset o Dataset normal
    if hasattr(dataset, 'datasets'):  # ConcatDataset
        for d in dataset.datasets:
            labels.extend([label for _, label in d])
    else:
        labels = [label for _, label in dataset]

    counter = Counter(labels)
    print(f"📊 Distribución de clases en {name}: {dict(counter)}")
    


def main():
    parser = argparse.ArgumentParser(description="Train a model for FAS classification")
    parser.add_argument('--input_dir', type=str, default="/mnt/d2/competicion")
    

    args = parser.parse_args()
    print(f"args:{args}")


    # train_ds_6 = DatasetFactory.create(
    #     num_classes=6,
    #     protocol_file=f"{args.input_dir}/Protocol-train.txt",
    #     root_dir=f"{args.input_dir}/Data-train",
    #     transform=None
    # )

    # get_or_compute_class_weights(train_ds, 2, weights_path="class_weights_2clases_aug.npy", force_recompute=True)


    # train_ds_15 = DatasetFactory.create(
    #     num_classes=15,
    #     protocol_file=f"{args.input_dir}/Protocol-train.txt",
    #     root_dir=f"{args.input_dir}/Data-train",
    #     transform=None
    # )

    # get_or_compute_class_weights(train_ds_15, 15,weights_path="class_weights_15clases.npy", force_recompute=True)


    train_ds = DatasetFactory.create(
            num_classes=2,
            protocol_file=f"{args.input_dir}/Protocol-train.txt",
            root_dir=f"{args.input_dir}/Data-train",
            transform=None
        )

    print_class_distribution(train_ds, name="Antes del balanceo")

    # Crear dataset de bonafides augmentados
    aug_bonafide_ds = load_augmented_bonafides(
        dir_path="/mnt/d2/competicion/Data-train-augmented",
        label=0,
        transform=None
    )

    train_ds = ConcatDataset([train_ds, aug_bonafide_ds])
    
    print_class_distribution(train_ds, name="Después del balanceo")

    get_or_compute_class_weights(train_ds, 2,weights_path="class_weights_2clases_aug.npy", force_recompute=True)

if __name__ == "__main__":
    main()
