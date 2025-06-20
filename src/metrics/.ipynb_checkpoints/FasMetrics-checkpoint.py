import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def evaluar_modelo(modelo, dataloader, num_clases, device="cuda", verbose=True, save_file=None):
    """
    Evalúa un modelo de clasificación binaria o ternaria (2 o 3 clases), asumiendo que la clase 0 es 'bonafide'.
    Calcula precisión, recall, pérdida promedio, AUC, EER, APCER, BPCER y ACER.
    Parámetros:
        - modelo: modelo de clasificación ya entrenado (por ejemplo, una red neuronal de PyTorch).
        - dataloader: iterable de evaluación que proporciona batches de (datos, etiquetas).
        - num_clases: 2 o 3, número de clases del modelo (clase 0 = bonafide).
        - device: dispositivo para ejecutar la evaluación ("cuda" por defecto, puede ser "cpu").
        - verbose: si True, imprime las métricas calculadas.
        - save_file: ruta de archivo CSV para guardar resultados (si se proporciona).
    Retorna:
        Un diccionario con los valores de las métricas calculadas.
    """
    # Validación de número de clases
    if num_clases not in [2, 3]:
        raise ValueError("La función soporta solo clasificación binaria o ternaria (2 o 3 clases).")
    modelo.to(device)
    modelo.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []

    # Iterar sobre el dataloader
    with torch.no_grad():
        for batch in dataloader:
            entradas, etiquetas = batch
            entradas = entradas.to(device)
            etiquetas = etiquetas.to(device)
            # Forward del modelo
            logits = modelo(entradas)
            # Cálculo de pérdida (entropía cruzada, suma sobre el batch)
            loss = F.cross_entropy(logits, etiquetas, reduction='sum')
            total_loss += loss.item()
            total_samples += etiquetas.size(0)
            # Probabilidad de clase 0 (bonafide) usando softmax
            probabilidades = F.softmax(logits, dim=1)
            prob_bonafide = probabilidades[:, 0]  # probabilidad predicha de ser bonafide
            # Almacenar para cálculo global
            all_probs.append(prob_bonafide.cpu().numpy())
            all_labels.append(etiquetas.cpu().numpy())

    # Concatenar todos los resultados
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    # Pérdida promedio en todo el conjunto
    perdida_promedio = total_loss / total_samples

    # Construir etiquetas binarias: 1 = bonafide (clase 0), 0 = ataque (clase != 0)
    y_true = (all_labels == 0).astype(np.int32)
    # Predicciones binarizadas con umbral 0.5 sobre probabilidad bonafide
    y_pred = (all_probs >= 0.5).astype(np.int32)

    # Calcular TP, FP, FN, TN
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    # Precisión y recall (evitando división por cero)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # AUC (área bajo la curva ROC) respecto a bonafide vs ataque
    auc = roc_auc_score(y_true, all_probs)

    # EER (Equal Error Rate) cálculo mediante la curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, all_probs, pos_label=1)
    fnr = 1 - tpr
    # Índice donde |FPR - FNR| es mínimo (aproximación al punto EER)
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2  # valor de EER

    # APCER, BPCER al umbral 0.5, y ACER correspondiente
    apcer = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # ataques clasificados como bonafide / total ataques
    bpcer = FN / (TP + FN) if (TP + FN) > 0 else 0.0  # bonafide clasificados como ataque / total bonafide
    acer = (apcer + bpcer) / 2.0

    # Imprimir métricas si verbose
    if verbose:
        print(f"Precisión: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Pérdida promedio: {perdida_promedio:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"EER: {eer:.4f}")
        print(f"APCER: {apcer:.4f}")
        print(f"BPCER: {bpcer:.4f}")
        print(f"ACER: {acer:.4f}")

    # Guardar resultados en CSV si se indicó una ruta
    if save_file:
        with open(save_file, "w", newline="") as f:
            # Fila de encabezados
            f.write("Precisión,Recall,Pérdida promedio,AUC,EER,APCER,BPCER,ACER\n")
            # Fila de valores (formato decimal con 4 cifras significativas)
            f.write(f"{precision:.4f},{recall:.4f},{perdida_promedio:.4f},{auc:.4f},"
                    f"{eer:.4f},{apcer:.4f},{bpcer:.4f},{acer:.4f}\n")

    # Devolver un diccionario con las métricas
    return {
        "precision": precision,
        "recall": recall,
        "loss_avg": perdida_promedio,
        "auc": auc,
        "eer": eer,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer
    }
