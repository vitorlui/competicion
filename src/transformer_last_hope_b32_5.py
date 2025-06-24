import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import vit_b_32, ViT_B_32_Weights
from PIL import Image
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from train.Eval import evaluate_model2
from ds.DatasetUtils import get_class_weights
from metrics.GenerateScores import generate_prediction_file2
from datetime import datetime
import argparse
from time import sleep
from train.Train import train_model
from ds.TransformFactory import TransformFactory
from models.ModelFactory import ModelFactory
from ds.DatasetFactory import DatasetFactory

class LiveFakeDataset(Dataset):
    def __init__(self, protocol_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(protocol_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                # Etiqueta binaria: 1 = Live, 0 = Fake
                binary_label = 1 if label == '0_0_0' else 0
                full_path = os.path.join(root_dir, os.path.basename(path))
                self.samples.append((full_path, binary_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cuda:1")  # segunda GPU (si existe)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

device = "cuda:1"

# Transformaciones
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


num_classes_iter = 4

# Dataset y loader
train_ds = LiveFakeDataset("Protocol-train.txt", "/mnt/d2/competicion/Data-train", transform=tfms)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=3)

torch.cuda.empty_cache()

# Modelo
weights = ViT_B_32_Weights.IMAGENET1K_V1
model = vit_b_32(weights=weights)
# model = vit_h_14()
# model = vit_b_16()

model.heads.head = nn.Linear(model.heads.head.in_features, 5)

# model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

model = model.to(device)

# Entrenamiento simple
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

output_dir = "ck_b32_5c_lasthope"
input_dir = "/mnt/d2/competicion"


experiment_time = datetime.now().strftime("%d%m_%H%M")
output_dir = f"outputs/{output_dir}_{experiment_time}"

os.makedirs(output_dir, exist_ok=True)

eval_ds = DatasetFactory.create(
    num_classes=2,
    protocol_file=f"{input_dir}/Protocol-val.txt",
    root_dir=f"{input_dir}/Data-val",
    transform=tfms,
    is3d= False
)

eval2_ds = DatasetFactory.create(
    num_classes=2,
    protocol_file=f"{input_dir}/Protocol-vt.txt",
    root_dir=f"{input_dir}/Data-vt",
    transform=tfms,
    is3d= False
)

eval_loader = DataLoader(eval_ds, batch_size=16)
eval2_loader = DataLoader(eval2_ds, batch_size=16)
metrics_log = []

# torch.set_float32_matmul_precision('medium')


model_name = "b32_5c"
checkpoint_every = 5
epochs = 100
num_classes = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed - Avg Loss: {train_loss:.4f}")

    row = {
            "Model": model_name,
            "Epoch": epoch,
            "Transform": "bilinear",
            "Loss": train_loss,
            "ACC": None,
            "Precision": None,
            "Recall": None,
            "AUC": None,
            "EER": None,
            "APCER": None,
            "BPCER": None,
            "ACER": None,
            "Split": "train"
        }

    # Evaluar y guardar checkpoint si toca
    if epoch % checkpoint_every == 0:
        ckpt_path = os.path.join(output_dir, f"{model_name}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint guardado: {ckpt_path}")

        train_metrics = evaluate_model2(model, train_loader, device, num_classes, desc="train")
        eval_metrics = evaluate_model2(model, eval_loader, device, num_classes, desc="eval")
        eval2_metrics = evaluate_model2(model, eval2_loader, device, num_classes, desc="vt")

        for split_name, metrics in zip(["val0", "val1", "val2"], [train_metrics, eval_metrics, eval2_metrics]):
            row_copy = row.copy()
            row_copy.update(metrics)
            row_copy["Split"] = split_name
            metrics_log.append(row_copy)

        # // add here the file generation to the same checkpoint directory
        # Generar el archivo de puntuaciones
        generate_prediction_file2(
            model=model,
            transform=tfms,
            model_name=model_name,
            transformer_name="bilinear",
            epoch=epoch,
            device=device,
            input_dir=input_dir,
            output_dir=output_dir               
        )
    else:
        metrics_log.append(row)

    # Guardar log CSV
    df = pd.DataFrame(metrics_log)
    csv_path = os.path.join(output_dir, f"{model_name}_training_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"Log de métricas guardado: {csv_path}")
