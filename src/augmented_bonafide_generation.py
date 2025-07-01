import os
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm

# 🧭 Configuración
INPUT_DIR = Path("/mnt/d2/competicion/Data-train")
PROTOCOL_FILE = Path("/mnt/d2/competicion/Protocol-train.txt")
BONAFIDE_PREFIX = "0_0_0"
OUTPUT_DIR = Path("/mnt/d2/competicion/Data-train-augmented")
AUGS_PER_IMAGE = 10  # Número de transformaciones por imagen

# 🧰 Transformaciones que aplicaremos
augmentation_transforms = [
    T.RandomRotation(degrees=15),
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip(p=1.0),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.GaussianBlur(kernel_size=5)
]

# 🏁 Crear directorio de salida
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 🔍 Leer solo paths bonafide
with open(PROTOCOL_FILE, "r") as f:
    bonafide_lines = [
        line.strip() for line in f.readlines()
        if line.strip().split()[1] == BONAFIDE_PREFIX
    ]

print(f"Encontradas {len(bonafide_lines)} imágenes bonafide para aumentar.")

# 🔁 Por cada imagen, aplicar augmentaciones
for line in tqdm(bonafide_lines):
    path_rel, label = line.strip().split()
    image_path = INPUT_DIR / Path(path_rel).name

    if not image_path.exists():
        continue

    img = Image.open(image_path).convert("RGB")
    base_name = image_path.stem

    for i in range(AUGS_PER_IMAGE):
        tfm = augmentation_transforms[i % len(augmentation_transforms)]
        aug_img = tfm(img)
        aug_name = f"{base_name}_aug{i}.png"
        aug_img.save(OUTPUT_DIR / aug_name)
