# Train.py

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from train.Eval import evaluate_model
from ds.DatasetUtils import get_class_weights
from metrics.GenerateScores import generate_prediction_file
from datetime import datetime
from torch.optim import AdamW


def train_model(
    model,
    train_loader,
    eval_loader,
    eval2_loader,
    transformer_name,
    tfms,
    model_name="model",
    num_classes=2,
    class_weights_path=None,
    checkpoint_every=5,
    epochs=15,
    device="cuda",
    input_dir=".",
    output_dir="checkpoints",
    hyperparams = {"lr": 1e-4, "weight_decay": 1e-4}   
):

    experiment_time = datetime.now().strftime("%d%m_%H%M")

    output_dir = f"outputs/{output_dir}_{experiment_time}"

    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            get_class_weights(class_weights_path),
            dtype=torch.float
        ).to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyperparams.get("lr", 1e-4), 
        weight_decay=hyperparams.get("weight_decay", 1e-4)
    )
    
    # criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    metrics_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            # print(f"🧪 imgs.shape: {imgs.shape}")  # Muestra la forma completa
            # print(f"🧪 imgs.ndim: {imgs.ndim}")    # Muestra el número de dimensiones

    
            # # Convertir de (B, F, C, H, W) → (B, C, T, H, W)
            # if imgs.ndim == 5:
            #     imgs = imgs.permute(0, 2, 1, 3, 4)

            if imgs.ndim == 5:
                if imgs.shape[1] != 3:
                    imgs = imgs.permute(0, 2, 1, 3, 4)
                
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} completed - Avg Loss: {train_loss:.4f}")

        row = {
            "Model": model_name,
            "Epoch": epoch,
            "Transform": transformer_name,
            "Loss": train_loss,
            "ACC": None,
            "Precision": None,
            "Recall": None,
            "AUC": None,
            "EER": None,
            "APCER": None,
            "BPCER": None,
            "ACER": None,
            "Split": "train"
        }

        # Evaluar y guardar checkpoint si toca
        if epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(output_dir, f"{model_name}_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint guardado: {ckpt_path}")

            train_metrics = evaluate_model(model, train_loader, device, num_classes, desc="train_eval")
            eval_metrics = evaluate_model(model, eval_loader, device, num_classes, desc="eval")
            eval2_metrics = evaluate_model(model, eval2_loader, device, num_classes, desc="vt")

            for split_name, metrics in zip(["val0","val1", "val2"], [eval_metrics, eval2_metrics]):
                row_copy = row.copy()
                row_copy.update(metrics)
                row_copy["Split"] = split_name
                metrics_log.append(row_copy)

            # // add here the file generation to the same checkpoint directory
            # Generar el archivo de puntuaciones
            generate_prediction_file(
                model=model,
                transform=tfms,
                model_name=model_name,
                transformer_name=transformer_name,
                epoch=epoch,
                device=device,
                input_dir=input_dir,
                output_dir=output_dir               
            )
        else:
            metrics_log.append(row)

        # Guardar log CSV
        df = pd.DataFrame(metrics_log)
        csv_path = os.path.join(output_dir, f"{model_name}_training_log.csv")
        df.to_csv(csv_path, index=False)
        print(f"Log de métricas guardado: {csv_path}")

    return model


def train_iteractive(
    model,
    train_loaders,  # AHORA ES UNA LISTA DE LOADER
    eval_loader,
    eval2_loader,
    transformer_name,
    tfms,
    model_name="model",
    num_classes=2,
    class_weights_path=None,
    checkpoint_every=5,
    epochs=15,
    device="cuda",
    input_dir=".",
    output_dir="checkpoints"
):
    experiment_time = datetime.now().strftime("%d%m_%H%M")
    output_dir = f"outputs/{output_dir}_{experiment_time}"
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            get_class_weights(class_weights_path),
            dtype=torch.float
        ).to(device)
    )

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    metrics_log = []

    for iter_idx, train_loader in enumerate(train_loaders):
        print(f"\n🌀 Entrenamiento iteración {iter_idx + 1}/{len(train_loaders)}")
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            loop = tqdm(train_loader, desc=f"[Iter {iter_idx+1}] Epoch {epoch}/{epochs}", leave=False)

            for imgs, labels in loop:
                imgs, labels = imgs.to(device), labels.to(device)

                if imgs.ndim == 5 and imgs.shape[1] != 3:
                    imgs = imgs.permute(0, 2, 1, 3, 4)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            train_loss = running_loss / len(train_loader)
            print(f"[Iter {iter_idx+1}] Epoch {epoch} completed - Avg Loss: {train_loss:.4f}")

            row = {
                "Model": model_name,
                "Epoch": epoch,
                "Transform": transformer_name,
                "Loss": train_loss,
                "ACC": None,
                "Precision": None,
                "Recall": None,
                "AUC": None,
                "EER": None,
                "APCER": None,
                "BPCER": None,
                "ACER": None,
                "Split": f"train_iter{iter_idx+1}"
            }

            # Evaluación y checkpoint
            if epoch % checkpoint_every == 0:
                ckpt_path = os.path.join(output_dir, f"{model_name}_iter{iter_idx+1}_epoch{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint guardado: {ckpt_path}")

                eval_metrics0 = evaluate_model(model, train_loader, device, num_classes, desc="eval0")
                eval_metrics = evaluate_model(model, eval_loader, device, num_classes, desc="eval")
                eval2_metrics = evaluate_model(model, eval2_loader, device, num_classes, desc="vt")

                for split_name, metrics in zip(["val0","val1", "val2"], [eval_metrics0, eval_metrics, eval2_metrics]):
                    row_copy = row.copy()
                    row_copy.update(metrics)
                    row_copy["Split"] = f"{split_name}_iter{iter_idx+1}"
                    metrics_log.append(row_copy)

                generate_prediction_file(
                    model=model,
                    transform=tfms,
                    model_name=model_name,
                    transformer_name=transformer_name,
                    epoch=epoch,
                    device=device,
                    input_dir=input_dir,
                    output_dir=output_dir
                )
            else:
                metrics_log.append(row)

            # Guardar CSV
            df = pd.DataFrame(metrics_log)
            csv_path = os.path.join(output_dir, f"{model_name}_training_log.csv")
            df.to_csv(csv_path, index=False)

    return model