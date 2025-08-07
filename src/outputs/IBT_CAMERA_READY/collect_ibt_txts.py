import os
import shutil
from pathlib import Path

# Ruta base
base_dir = Path(".")
output_dir = Path("ibt_txts")
output_dir.mkdir(parents=True, exist_ok=True)

# Recorre cada subcarpeta en IBT_CAMERA_READY
for folder in base_dir.iterdir():
    if folder.is_dir():
        folder_name = folder.name

        # Determina el sufijo por nombre de carpeta
        suffix = ""
        if "_pt0" in folder_name:
            suffix = "_pt0"
        elif "_pt1" in folder_name:
            suffix = "_pt1"
        else:
            continue  # ignora carpetas que no contienen pt0 o pt1

        # Buscar todos los .txt dentro de la subcarpeta
        for txt_file in folder.glob("*.txt"):
            new_name = txt_file.stem + suffix + ".txt"
            destination = output_dir / new_name
            shutil.copy(txt_file, destination)
            print(f"Copied: {txt_file.name} → {destination.name}")
