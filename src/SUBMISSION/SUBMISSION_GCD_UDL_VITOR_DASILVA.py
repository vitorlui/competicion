import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from torchvision.models import  vit_b_16
from tqdm import tqdm
from datetime import datetime
from torch.optim import AdamW
from tqdm import tqdm
from IPython.display import display
from pathlib import Path
from typing import Callable, Optional
from torch.utils.data import Dataset
from PIL import Image
from typing import List
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, roc_curve
)
from typing import Callable, Optional, Union
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class ValDataset(Dataset):
    def __init__(self, protocol_file, transform=None, input_dir ="."):
        self.image_paths = []
        self.transform = transform

        with open(f"{input_dir}/{protocol_file}", 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    self.image_paths.append(f"{input_dir}/{path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path
    
def generate_prediction_file(model, transform, model_name, transformer_name, epoch, device="cuda", input_dir=".", output_dir="."):
    model.eval()

    
    # Rutas a protocolos
    val_ds = ValDataset("Protocol-val-notlabeled.txt", transform=transform,  input_dir=input_dir)
    val_loader = DataLoader(val_ds, batch_size=96, shuffle=False)

    test_ds = ValDataset("Protocol-test.txt", transform=transform,  input_dir=input_dir)
    test_loader = DataLoader(test_ds, batch_size=96, shuffle=False)

    results = []

    with torch.no_grad():
        for loader, split_name in [(val_loader, "val"), (test_loader, "test")]:
            for imgs, paths in tqdm(loader, desc=f"Evaluando {split_name}"):
                # Convertir de (B, F, C, H, W) → (B, C, T, H, W)
                # if imgs.ndim == 5:
                #     imgs = imgs.permute(0, 2, 1, 3, 4)

                if imgs.ndim == 5:
                    if imgs.shape[1] != 3:
                        # print("PERMUTED")
                        imgs = imgs.permute(0, 2, 1, 3, 4)

                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                live_probs = probs[:, 0].cpu().numpy()
                results.extend(zip(paths, live_probs))

    experiment_name = f"{model_name}_epoch{epoch}_{transformer_name}"
    output_file = f"{output_dir}/phase2_score_{experiment_name}.txt"

    with open(output_file, 'w') as f:
        for path, score in results:
            f.write(f"{path} {score:.5f}\n")

    print(f"Generated File: {output_file}")

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
    


class TwoClassDatasetCustom(Dataset):
    """
    Binary classification:
        0 -> bonafide (e.g., ["0_", "0_1", "0_2"])
        1 -> fraud (e.g., ["1_", "2_", "3_1"])
    """

    def __init__(
        self,
        protocol_file: str,
        root_dir: Union[str, Path],
        bonafide_prefixes: List[str],
        fraud_prefixes: List[str],
        transform: Optional[Callable] = None
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        with open(protocol_file, "r") as f:
            for line in f:
                
                path, label = line.strip().split()
                prefix = label.split("_", 1)[0] + "_"  # keep the base prefix (e.g., "0_")

                full_path = self.root_dir / Path(path).name

                if prefix in bonafide_prefixes:
                    target = 0
                elif prefix in fraud_prefixes:
                    # print(line)
                    target = 1
                else:
                    continue  # ignore classes not listed

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
#    Dataset Factory
# ========================
class DatasetFactory:
    @staticmethod
    def create(num_classes: int
               , protocol_file: str
               , root_dir: str
               , transform: Optional[Callable] = None
               , is3d: bool = False
               , apply_filters: bool = True
               , preproc: bool = False
               , bonafide_prefixes: List[str] = None
               , fraud_prefixes: List[str]= None
                ) -> Dataset:
        
        if num_classes == 2:
            return TwoClassDataset(protocol_file, root_dir, transform)
        elif num_classes == 21:
            return TwoClassDatasetCustom(protocol_file, root_dir, bonafide_prefixes, fraud_prefixes, transform=transform)
        # elif num_classes == 3:
        #     return ThreeClassDataset(protocol_file, root_dir, transform)
        # elif num_classes == 6:
        #     return SixClassDataset(protocol_file, root_dir, transform)
        # elif num_classes == 15:
        #     return FifteenClassDataset(protocol_file, root_dir, transform)
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")


class BaseModelWrapper(nn.Module):
    def __init__(self, model_fn, output_attr, num_classes, pretrained=False):
        super().__init__()
        self.model = model_fn(weights="IMAGENET1K_V1" if pretrained else None)
        self._adjust_output_layer(output_attr, num_classes)

    def _adjust_output_layer(self, output_attr, num_classes):
        module = self.model
        for part in output_attr[:-1]:
            module = getattr(module, part)
        last_name = output_attr[-1]
        last_layer = getattr(module, last_name)
        new_layer = nn.Linear(last_layer.in_features, num_classes)
        setattr(module, last_name, new_layer)

    def forward(self, x):
        return self.model(x)

class ViTB16(BaseModelWrapper):
    def __init__(self, num_classes, pretrained):
        super().__init__(vit_b_16, ["heads", "head"], num_classes, pretrained)

#For submission I`ve included only the best model b16
class ModelFactory:
    MODEL_MAP = {
        # "vit_b_32": ViTB32,
        "vit_b_16": ViTB16,
        # "resnet50": ResNet50,
        # "resnet152": ResNet152,
        # "resnext50_32x4d": ResNeXt50,
        # "resnext101_64x4d": ResNeXt101,
        # "r2plus1d_18": R2Plus1D18
    }

    @staticmethod
    def list_model_names():
        return list(ModelFactory.MODEL_MAP.keys())

    @staticmethod
    def create(model_name: str, num_classes: int, device="cpu", pretrained=False, multiGPU=False):
        print(f"ModelFactory->create: Pretrained: {pretrained}")
        if model_name not in ModelFactory.MODEL_MAP:
            raise ValueError(f"Modelo «{model_name}» no está soportado.")
        model_cls = ModelFactory.MODEL_MAP[model_name]
        # model = model_cls(num_classes)
        model = model_cls(num_classes, pretrained=pretrained)

        if multiGPU:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
            model = torch.nn.DataParallel(model)
            
        return model.to(device)
    

def evaluate_model(
    model,
    dataloader,
    device="cuda",
    num_classes=2,
    desc = "Evaluating",
    verbose=True
):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Convertir de (B, F, C, H, W) → (B, C, T, H, W)
            if imgs.ndim == 5:
                imgs = imgs.permute(0, 2, 1, 3, 4)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 0]  # prob. de bonafide
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    y_true_bin = (y_true == 0).astype(int)

    TP = np.sum((y_pred == 0) & (y_true == 0))
    TN = np.sum((y_pred != 0) & (y_true != 0))
    FP = np.sum((y_pred == 0) & (y_true != 0))
    FN = np.sum((y_pred != 0) & (y_true == 0))

    precision = precision_score(y_true_bin, (y_probs >= 0.5).astype(int))
    recall = recall_score(y_true_bin, (y_probs >= 0.5).astype(int))
    auc_score = roc_auc_score(y_true_bin, y_probs)
    fpr, tpr, _ = roc_curve(y_true_bin, y_probs)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

    apcer = FP / (FP + TN + 1e-8)
    bpcer = FN / (FN + TP + 1e-8)
    acer = (apcer + bpcer) / 2

    metrics = {
        "Loss": total_loss / len(dataloader),
        "ACC": np.mean(y_pred == y_true),
        "Precision": precision,
        "Recall": recall,
        "AUC": auc_score,
        "EER": eer,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
    }
    
    if verbose:
        print("\n Evaluation:")
        df_metrics = pd.DataFrame([metrics])
        display(df_metrics.round(4))

    return metrics
    

def get_class_weights(weights_path="class_weights.npy"):
    """
    Carga pesos de clases balanceados para un dataset de clasificación.

    Parámetros:
        dataset: PyTorch Dataset
            Dataset que retorna (imagen, label)
        weights_path: str
            Ruta del archivo .npy donde guardar/cargar los pesos
        force_recompute: bool
            Si True, fuerza recalcular incluso si el archivo existe

    Retorna:
        class_weights: np.ndarray
            Vector de pesos (float) de tamaño [num_clases]
    """
    if os.path.exists(weights_path):
        class_weights = np.load(weights_path)
        print(f"Pesos cargados desde: {weights_path}")
        return class_weights


def train_iteractive(
    model,
    train_loaders,  # It is a loader list
    eval_loader,
    transformer_name,
    tfms,
    model_name="model",
    num_classes=2,
    class_weights_path=None,
    checkpoint_every=5,
    epochs=15,
    device="cuda",
    input_dir=".",
    output_dir="checkpoints"
):
    experiment_time = datetime.now().strftime("%d%m_%H%M")
    output_dir = f"outputs/{output_dir}_{experiment_time}"
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            get_class_weights(class_weights_path),
            dtype=torch.float
        ).to(device)
    )

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    metrics_log = []

    for iter_idx, train_loader in enumerate(train_loaders):
        print(f"\n Training Iteraction {iter_idx + 1}/{len(train_loaders)}")
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            loop = tqdm(train_loader, desc=f"[Iter {iter_idx+1}] Epoch {epoch}/{epochs}", leave=False)

            for imgs, labels in loop:
                imgs, labels = imgs.to(device), labels.to(device)

                if imgs.ndim == 5 and imgs.shape[1] != 3:
                    imgs = imgs.permute(0, 2, 1, 3, 4)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            train_loss = running_loss / len(train_loader)
            print(f"[Iter {iter_idx+1}] Epoch {epoch} completed - Avg Loss: {train_loss:.4f}")

            row = {
                "Model": model_name,
                "Epoch": epoch,
                "Transform": transformer_name,
                "Loss": train_loss,
                "ACC": None,
                "Precision": None,
                "Recall": None,
                "AUC": None,
                "EER": None,
                "APCER": None,
                "BPCER": None,
                "ACER": None,
                "Split": f"train_iter{iter_idx+1}"
            }

            # Evaluation and checkpoint
            if epoch % checkpoint_every == 0:
                ckpt_path = os.path.join(output_dir, f"{model_name}_iter{iter_idx+1}_epoch{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint guardado: {ckpt_path}")

                eval_metrics = evaluate_model(model, eval_loader, device, num_classes, desc="eval")

                for split_name, metrics in zip(["val1"], [eval_metrics]):
                    row_copy = row.copy()
                    row_copy.update(metrics)
                    row_copy["Split"] = f"{split_name}_iter{iter_idx+1}"
                    metrics_log.append(row_copy)

                generate_prediction_file(
                    model=model,
                    transform=tfms,
                    model_name=model_name,
                    transformer_name=transformer_name,
                    epoch=epoch,
                    device=device,
                    input_dir=input_dir,
                    output_dir=output_dir
                )
            else:
                metrics_log.append(row)

            # Save CSV
            df = pd.DataFrame(metrics_log)
            csv_path = os.path.join(output_dir, f"{model_name}_training_log.csv")
            df.to_csv(csv_path, index=False)

    return model
    
def main():
    parser = argparse.ArgumentParser(description="Train a model for FAS classification")
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g. vit_b_32, resnet50)')
    parser.add_argument('--transformer', type=str, required=True, help='Transformer name from TransformFactory')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device (default: cuda:0)')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--every_epoch', type=int, default=15, help='Checkpoint frequency (default: 5)')
    parser.add_argument('--classes', type=int, default=2, help='Number of output classes (2 or 3)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--preproc', type=bool, default=False)
    parser.add_argument('--multiGPU', type=bool, default=False)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--input_dir', type=str, default=".")
    parser.add_argument('--label', type=str, default=None)
    
    label = "iteract"
    args = parser.parse_args()
    # print(f"args:{args}")
    transform_name = args.transformer
    num_classes_iter = args.classes

    tfms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean =  [0.4603, 0.3696, 0.3388]
                                                         , std = [0.2246, 0.2059, 0.1994]
                                   )])

    print("Transformation:", transform_name)    

    # Human-readable label mapping
    label_map = {
        "0_0_0": "Live Face",
        "1_0_0": "Print (2D Attack)",
        "1_0_1": "Replay (2D Attack)",
        "1_0_2": "Cutouts (2D Attack)",
        "1_1_0": "Transparent (3D Attack)",
        "1_1_1": "Plaster (3D Attack)",
        "1_1_2": "Resin (3D Attack)",
        "2_0_0": "Attribute-Edit (Digital Manipulation)",
        "2_0_1": "Face-Swap (Digital Manipulation)",
        "2_0_2": "Video-Driven (Digital Manipulation)",
        "2_1_0": "Pixel-Level (Digital Adversarial)",
        "2_1_1": "Semantic-Level (Digital Adversarial)",
        "2_2_0": "ID_Consistent (Digital Generation)",
        "2_2_1": "Style (Digital Generation)",
        "2_2_2": "Prompt (Digital Generation)"
    }


    # Observed training distribution per class
    class_distribution = {
        "2_1_0": 8364,
        "2_0_1": 6160,
        "2_1_1": 3757,
        "2_0_2": 1540,
        "2_0_0": 1476,
        "0_0_0": 839,
        "1_0_1": 109,
        "1_0_2": 79,
        "1_0_0": 43,
        # clases con 0 no se incluyen
    }

    # Sort spoofing classes by descending frequency, excluding bonafide
    ordered_fraud_labels = OrderedDict(
        sorted(
            {k: v for k, v in class_distribution.items() if k != "0_0_0"}.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    # --- Iterative Training Strategy ---
    # Fixed bonafide class for all iterations
    bonafide_label = "0_0_0"
    bonafide_prefixes = [bonafide_label.split("_")[0] + "_"]
    datasets_iterativos = []

    # Create binary datasets: bonafide vs. one spoofing class per iteration
    for i, (fraud_label, _) in enumerate(ordered_fraud_labels.items()):
        fraud_prefixes = [fraud_label.split("_")[0] + "_"]

        print(f"[Iteration {i+1}] Bonafide: {label_map[bonafide_label]} ({bonafide_prefixes})")
        print(f"Fraud: {label_map[fraud_label]} ({fraud_prefixes})")

        train_ds = DatasetFactory.create(
            num_classes=21,
            protocol_file=f"{args.input_dir}/Protocol-train.txt",
            root_dir=f"{args.input_dir}/Data-train",
            transform=tfms,
            is3d=False,
            bonafide_prefixes=bonafide_prefixes,
            fraud_prefixes=fraud_prefixes
        )

        datasets_iterativos.append((fraud_label, train_ds))
        break
    
    print("Number of binary training datasets:", len(datasets_iterativos))

    train_ds_all = DatasetFactory.create(
            num_classes=2,
            protocol_file=f"{args.input_dir}/Protocol-train.txt",
            root_dir=f"{args.input_dir}/Data-train",
            transform=tfms,
            is3d=False,
            bonafide_prefixes=bonafide_prefixes,
            fraud_prefixes=fraud_prefixes
        )

    datasets_iterativos.append(("all", train_ds_all))

    print("Total datasets including final full iteration:", len(datasets_iterativos))
    
    eval_ds = DatasetFactory.create(
        num_classes=2,
        protocol_file=f"{args.input_dir}/Protocol-val.txt",
        root_dir=f"{args.input_dir}/Data-val",
        transform=tfms,
        is3d= False
    )

    model = ModelFactory.create(args.model, num_classes=num_classes_iter, device=args.device, pretrained=args.pretrained, multiGPU=args.multiGPU)

     # Create a DataLoader for each binary iteration dataset
    train_dataloaders = [
                            DataLoader(
                                dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2,
                            )
                            for _, dataset in datasets_iterativos
                        ]
    
    eval_loader = DataLoader(eval_ds, batch_size=16)

     # Train using the iterative binary classification strategy
    model = train_iteractive(
        model=model,
        train_loaders=train_dataloaders, 
        eval_loader=eval_loader,
        transformer_name=transform_name,
        tfms=tfms,
        model_name=f"{args.model}_{num_classes_iter}c",
        num_classes=num_classes_iter,
        class_weights_path=f"class_weights_{num_classes_iter}clases.npy",
        checkpoint_every=args.every_epoch,
        epochs=args.epochs,
        device=args.device,
        input_dir=args.input_dir,
        output_dir=f"ck_{args.model}_{num_classes_iter}_pt{1 if args.pretrained else 0}_{transform_name}{label}"
    )


if __name__ == "__main__":
    main()