# Eval.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, roc_curve
)
import pandas as pd
from IPython.display import display

def evaluate_model(
    model,
    dataloader,
    device="cuda",
    num_classes=2,
    desc = "Evaluating",
    verbose=True
):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Convertir de (B, F, C, H, W) → (B, C, T, H, W)
            if imgs.ndim == 5:
                imgs = imgs.permute(0, 2, 1, 3, 4)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 0]  # prob. de bonafide
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    y_true_bin = (y_true == 0).astype(int)

    TP = np.sum((y_pred == 0) & (y_true == 0))
    TN = np.sum((y_pred != 0) & (y_true != 0))
    FP = np.sum((y_pred == 0) & (y_true != 0))
    FN = np.sum((y_pred != 0) & (y_true == 0))

    precision = precision_score(y_true_bin, (y_probs >= 0.5).astype(int))
    recall = recall_score(y_true_bin, (y_probs >= 0.5).astype(int))
    auc_score = roc_auc_score(y_true_bin, y_probs)
    fpr, tpr, _ = roc_curve(y_true_bin, y_probs)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

    apcer = FP / (FP + TN + 1e-8)
    bpcer = FN / (FN + TP + 1e-8)
    acer = (apcer + bpcer) / 2

    metrics = {
        "Loss": total_loss / len(dataloader),
        "ACC": np.mean(y_pred == y_true),
        "Precision": precision,
        "Recall": recall,
        "AUC": auc_score,
        "EER": eer,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
    }

    # if verbose:
    #     print("\n📊 Métricas de evaluación:")
    #     for k, v in metrics.items():
    #         print(f"{k}: {v:.4f}")
    
    if verbose:
        print("\n Métricas de evaluación:")
        df_metrics = pd.DataFrame([metrics])
        display(df_metrics.round(4))

    return metrics

def evaluate_model2(
    model,
    dataloader,
    device="cuda",
    num_classes=2,
    desc = "Evaluating",
    verbose=True
):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Convertir de (B, F, C, H, W) → (B, C, T, H, W)
            if imgs.ndim == 5:
                imgs = imgs.permute(0, 2, 1, 3, 4)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 1]  # prob. de bonafide
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    y_true_bin = (y_true == 0).astype(int)

    TP = np.sum((y_pred == 0) & (y_true == 0))
    TN = np.sum((y_pred != 0) & (y_true != 0))
    FP = np.sum((y_pred == 0) & (y_true != 0))
    FN = np.sum((y_pred != 0) & (y_true == 0))

    precision = precision_score(y_true_bin, (y_probs >= 0.5).astype(int))
    recall = recall_score(y_true_bin, (y_probs >= 0.5).astype(int))
    auc_score = roc_auc_score(y_true_bin, y_probs)
    fpr, tpr, _ = roc_curve(y_true_bin, y_probs)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

    apcer = FP / (FP + TN + 1e-8)
    bpcer = FN / (FN + TP + 1e-8)
    acer = (apcer + bpcer) / 2

    metrics = {
        "Loss": total_loss / len(dataloader),
        "ACC": np.mean(y_pred == y_true),
        "Precision": precision,
        "Recall": recall,
        "AUC": auc_score,
        "EER": eer,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
    }

    # if verbose:
    #     print("\n📊 Métricas de evaluación:")
    #     for k, v in metrics.items():
    #         print(f"{k}: {v:.4f}")
    
    if verbose:
        print("\n Métricas de evaluación:")
        df_metrics = pd.DataFrame([metrics])
        display(df_metrics.round(4))

    return metrics

