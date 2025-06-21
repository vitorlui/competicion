import argparse
from time import sleep
from train.Train import train_model
from ds.TransformFactory import TransformFactory
from models.ModelFactory import ModelFactory
from ds.DatasetFactory import DatasetFactory
from torch.utils.data import DataLoader
from ds.DatasetUtils import get_or_compute_class_weights


def main():
    parser = argparse.ArgumentParser(description="Train a model for FAS classification")
    parser.add_argument('--input_dir', type=str, default="/mnt/d2/competicion")
    

    args = parser.parse_args()
    print(f"args:{args}")


    train_ds_6 = DatasetFactory.create(
        num_classes=6,
        protocol_file=f"{args.input_dir}/Protocol-train.txt",
        root_dir=f"{args.input_dir}/Data-train",
        transform=None
    )

    get_or_compute_class_weights(train_ds_6, 6, weights_path="class_weights_6clases.npy", force_recompute=True)


    train_ds_15 = DatasetFactory.create(
        num_classes=15,
        protocol_file=f"{args.input_dir}/Protocol-train.txt",
        root_dir=f"{args.input_dir}/Data-train",
        transform=None
    )

    get_or_compute_class_weights(train_ds_15, 15,weights_path="class_weights_15clases.npy", force_recompute=True)


if __name__ == "__main__":
    main()
