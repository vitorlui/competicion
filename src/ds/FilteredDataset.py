import os
from pathlib import Path
from typing import Callable, Optional, Union
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FilteredDataset(Dataset):
    def __init__(
        self,
        protocol_file: Union[str, Path],
        root_dir: Union[str, Path],
        apply_filters: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.apply_filters = apply_filters
        self.samples: list[tuple[str, int]] = []

        with open(protocol_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                # print(f"FilteredDataset line: {path}")
                prefix = label.split("_", 1)[0] + "_"
                target = 0 if prefix == "0_" else 1
                full_path = self.root_dir / Path(path).name
                self.samples.append((str(full_path), target))                

    def __len__(self):
        return len(self.samples)

    def _apply_filters(self, img_pil: Image.Image):
        filtered_images = []

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Filtro 0: Color Jitter
        jittered = ImageEnhance.Color(img_pil).enhance(2.0)
        filtered_images.append(jittered)

        # Filtro 1: Gaussian Blur
        blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=2))
        filtered_images.append(blurred)

        # Filtro 2: High-pass
        low_pass = cv2.GaussianBlur(img_cv, (9, 9), 0)
        high_pass = cv2.addWeighted(img_cv, 1.5, low_pass, -0.5, 0)
        filtered_images.append(Image.fromarray(cv2.cvtColor(high_pass, cv2.COLOR_BGR2RGB)))

        # Filtro 3: Low-pass
        filtered_images.append(Image.fromarray(cv2.cvtColor(low_pass, cv2.COLOR_BGR2RGB)))

        # Filtro 4: Histogram Equalization (Y channel)
        img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        filtered_images.append(Image.fromarray(img_eq))

        # Filtro 5: Grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        filtered_images.append(Image.fromarray(gray_rgb))

        # Filtro 6: Gradient Magnitude (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        magnitude_rgb = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
        filtered_images.append(Image.fromarray(magnitude_rgb))

        # Filtro 7: High Saturation Mask
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        _, sat_mask = cv2.threshold(s, 200, 255, cv2.THRESH_BINARY)
        sat_mask_rgb = cv2.cvtColor(sat_mask, cv2.COLOR_GRAY2RGB)
        filtered_images.append(Image.fromarray(sat_mask_rgb))

        return filtered_images

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        images = [img]  # original frame

        if self.apply_filters:
            images.extend(self._apply_filters(img))

        # aplicar transformaciones
        if self.transform:
            images = [self.transform(im) for im in images]

        # Stack de imágenes simulando video: (frames, C, H, W)
        video_tensor = torch.stack(images)  # ahora directamente tensores
        return video_tensor, label