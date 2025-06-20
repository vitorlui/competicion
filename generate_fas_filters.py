import os
import cv2
import numpy as np
from PIL import Image

def apply_filters(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Nombre base
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)

    # Leer imagen en RGB
    pil_img = Image.open(input_path).convert("RGB")
    img = np.array(pil_img)

    # f0 - Imagen original
    pil_img.save(os.path.join(output_dir, f"{name}_f0.png"))

    # f1 - Blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    Image.fromarray(blur).save(os.path.join(output_dir, f"{name}_f1.png"))

    # f2 - CLAHE (mejor para brillo/saturación)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    Image.fromarray(clahe_img).save(os.path.join(output_dir, f"{name}_f2.png"))

    # f3 - Edge detection (Canny)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    Image.fromarray(edge_rgb).save(os.path.join(output_dir, f"{name}_f3.png"))

    # f4 - High-frequency (original - blur)
    highpass = cv2.subtract(img, blur)
    Image.fromarray(highpass).save(os.path.join(output_dir, f"{name}_f4.png"))

    print(f"✅ Imágenes generadas en: {output_dir}")

    

# ====================
# Uso directo
# ====================
if __name__ == "__main__":
    input_path = "./src/Data-train/00000.png"   # <- Cambia esto
    output_dir = "salidas_filtros"             # <- Cambia esto
    apply_filters(input_path, output_dir)
