import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def compute_mean_std(dataset, batch_size=64, num_workers=2):
    """
    Calcula la media y desviación estándar por canal (RGB) de un dataset.

    Parámetros:
        dataset: torch.utils.data.Dataset
            Dataset que retorna imágenes como tensores (C x H x W).
        batch_size: int
            Tamaño de los lotes para el DataLoader.
        num_workers: int
            Número de procesos de carga en paralelo.

    Retorna:
        (mean, std): tuplas de listas float con 3 valores (uno por canal RGB)
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0

    for data, _ in tqdm(loader, desc="Computing mean/std"):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean.tolist(), std.tolist()



def get_or_compute_class_weights(dataset, weights_path="class_weights.npy", force_recompute=False):
    """
    Calcula o carga pesos de clases balanceados para un dataset de clasificación.

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
    if os.path.exists(weights_path) and not force_recompute:
        class_weights = np.load(weights_path)
        print(f"✅ Pesos cargados desde: {weights_path}")
        return class_weights

    # Extraer etiquetas
    labels = [label for _, label in dataset]
    classes = np.unique(labels)

    # Calcular pesos balanceados
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=np.array(labels)
    )

    # Guardar a disco
    np.save(weights_path, class_weights)
    print(f"✅ Pesos calculados y guardados en: {weights_path}")
    return class_weights

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
        print(f"✅ Pesos cargados desde: {weights_path}")
        return class_weights
