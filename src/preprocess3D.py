import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def get_transform():
    resize = 256
    interpolation = InterpolationMode.BICUBIC
    tfms = [
        transforms.Resize(resize, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
    composed = transforms.Compose(tfms)
    name = f"Resize{resize}bicubic_crop224"
    return name, composed


def apply_all_filters(img_pil: Image.Image):
    filtered_images = []
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Filtro 0: Color Jitter
    filtered_images.append(ImageEnhance.Color(img_pil).enhance(2.0))

    # Filtro 1: Gaussian Blur
    filtered_images.append(img_pil.filter(ImageFilter.GaussianBlur(radius=2)))

    # Filtro 2: High-pass
    low_pass = cv2.GaussianBlur(img_cv, (9, 9), 0)
    high_pass = cv2.addWeighted(img_cv, 1.5, low_pass, -0.5, 0)
    filtered_images.append(Image.fromarray(cv2.cvtColor(high_pass, cv2.COLOR_BGR2RGB)))

    # Filtro 3: Low-pass
    filtered_images.append(Image.fromarray(cv2.cvtColor(low_pass, cv2.COLOR_BGR2RGB)))

    # Filtro 4: Histogram Equalization (Y channel)
    img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    filtered_images.append(Image.fromarray(eq))

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


def main():
    input = "Data-test"
    input_dir = Path(input)  # carpeta con imágenes originales
    output_name, tfms = get_transform()
    output_dir = Path(f"{input}3D")
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(sorted(input_dir.glob("*.png")), desc="Generando filtros"):
        img = Image.open(image_path).convert("RGB")
        base_name = image_path.stem  # sin extensión

        # Guardar original transformado
        tensor = tfms(img)
        img_out = transforms.ToPILImage()(tensor)
        img_out.save(output_dir / f"{base_name}.png")

        # Aplicar filtros y guardar
        filtered = apply_all_filters(img)
        for i, f_img in enumerate(filtered):
            t = tfms(f_img)
            out_img = transforms.ToPILImage()(t)
            out_img.save(output_dir / f"{base_name}_f{i}.png")


if __name__ == "__main__":
    main()