import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

class TransformFactory:
    @staticmethod
    def get_preprocess_transforms():
        resize = 256
        crop = True
        # interpolation = InterpolationMode.BILINEAR
        interpolation = InterpolationMode.BICUBIC

        tfms = [transforms.Resize(resize, interpolation=interpolation)]
        if crop:
            tfms.append(transforms.CenterCrop(224))
        tfms.append(transforms.ToTensor())  # NO Normalize

        composed = transforms.Compose(tfms)
        name = f"Resize{resize}bicubic_crop224"
        return name, composed

# Carpeta de entrada
input_dir = Path("Data-train")
output_name, tfms = TransformFactory.get_preprocess_transforms()

# Carpeta de salida
output_dir = Path(f"PrepProc{output_name}")
output_dir.mkdir(parents=True, exist_ok=True)

# Protocolo a generar
protocol_path = Path(f"Protocol_{output_name}.txt")
protocol_lines = []

# Procesar imágenes
for image_path in tqdm(sorted(input_dir.glob("*.png")), desc="Procesando"):
    img = Image.open(image_path).convert("RGB")
    tensor = tfms(img)
    img_to_save = transforms.ToPILImage()(tensor)

    new_path = output_dir / image_path.name
    img_to_save.save(new_path)