# generate_scores.py
import os
import torch
from pathlib import Path
from typing import Callable, Optional, Union
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from models.ModelFactory import is_3d_model
from ds.DatasetFactory import DatasetFactory
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

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

class ValDataset3D(Dataset):
    def __init__(
        self,
        protocol_file: Union[str, Path],
        transform: Optional[Callable] = None,
        input_dir: str = ".",
        apply_filters: bool = True
    ):
        self.transform = transform
        self.apply_filters = apply_filters
        self.image_paths = []

        with open(f"{input_dir}/{protocol_file}", 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    self.image_paths.append(f"{input_dir}/{path}")

    def __len__(self):
        return len(self.image_paths)

    def _apply_filters(self, img_pil: Image.Image):
        filtered_images = []
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Filtros como en FilteredDataset
        filtered_images.append(ImageEnhance.Color(img_pil).enhance(2.0))  # Color Jitter
        filtered_images.append(img_pil.filter(ImageFilter.GaussianBlur(radius=2)))  # Blur

        low_pass = cv2.GaussianBlur(img_cv, (9, 9), 0)
        high_pass = cv2.addWeighted(img_cv, 1.5, low_pass, -0.5, 0)
        filtered_images.append(Image.fromarray(cv2.cvtColor(high_pass, cv2.COLOR_BGR2RGB)))  # High-pass
        filtered_images.append(Image.fromarray(cv2.cvtColor(low_pass, cv2.COLOR_BGR2RGB)))  # Low-pass

        img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        filtered_images.append(Image.fromarray(eq))  # Hist. Eq.

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        filtered_images.append(Image.fromarray(gray_rgb))  # Grayscale

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        mag_rgb = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
        filtered_images.append(Image.fromarray(mag_rgb))  # Gradient

        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        _, sat_mask = cv2.threshold(s, 200, 255, cv2.THRESH_BINARY)
        sat_mask_rgb = cv2.cvtColor(sat_mask, cv2.COLOR_GRAY2RGB)
        filtered_images.append(Image.fromarray(sat_mask_rgb))  # High Sat.

        return filtered_images

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        frames = [img]
        if self.apply_filters:
            frames.extend(self._apply_filters(img))

        if self.transform:
            frames = [self.transform(f) for f in frames]

        video_tensor = torch.stack(frames)  # (F, C, H, W)

        return video_tensor, img_path




def generate_prediction_file(model, transform, model_name, transformer_name, epoch, device="cuda", input_dir=".", output_dir="."):
    model.eval()

    print(f"ENTRAMOS generate_prediction_file {model_name}")

    if(is_3d_model(model_name)):
        print("\n\n ****** GENERATE SCORE: ENTRAMOS EN 3D ****** \n\n")
         # Rutas a protocolos
        val_ds = ValDataset3D("Protocol-val2.txt", transform=transform, input_dir=input_dir)
        val_loader = DataLoader(val_ds, batch_size=96, shuffle=False)

        test_ds = ValDataset3D("Protocol-test.txt", transform=transform,  input_dir=input_dir)
        test_loader = DataLoader(test_ds, batch_size=96, shuffle=False)

    else:
        
        # Rutas a protocolos
        val_ds = ValDataset("Protocol-val2.txt", transform=transform,  input_dir=input_dir)
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

    print(f"✅ Archivo generado: {output_file}")