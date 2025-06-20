# generate_scores.py

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import os

class ValDataset(Dataset):
    def __init__(self, protocol_file, transform=None):
        self.image_paths = []
        self.transform = transform

        with open(protocol_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    self.image_paths.append(path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path


def generate_prediction_file(model, transform, model_name, transformer_name, epoch, device="cuda", output_dir="."):
    model.eval()

    # Rutas a protocolos
    val_ds = ValDataset("Protocol-val2.txt", transform=transform)
    val_loader = DataLoader(val_ds, batch_size=96, shuffle=False)

    test_ds = ValDataset("Protocol-test.txt", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=96, shuffle=False)

    results = []

    with torch.no_grad():
        for loader, split_name in [(val_loader, "val"), (test_loader, "test")]:
            for imgs, paths in tqdm(loader, desc=f"Evaluando {split_name}"):
                # Convertir de (B, F, C, H, W) → (B, C, T, H, W)
                if imgs.ndim == 5:
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

    print(f"✅ Archivo generado: {output_file}")
