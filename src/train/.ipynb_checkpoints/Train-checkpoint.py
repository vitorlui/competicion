# Train.py

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from train.Eval import evaluate_model
from ds.DatasetUtils import get_class_weights
from metrics.GenerateScores import generate_prediction_file

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
    checkpoint_every=10,
    epochs=50,
    device="cuda",
    output_dir="checkpoints"
):

    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            get_class_weights(class_weights_path),
            dtype=torch.float
        ).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
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

            eval_metrics = evaluate_model(model, eval_loader, device, num_classes, desc="eval")
            eval2_metrics = evaluate_model(model, eval2_loader, device, num_classes, desc="vt")

            for split_name, metrics in zip(["val1", "val2"], [eval_metrics, eval2_metrics]):
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
                device=device
            )
        else:
            metrics_log.append(row)

    # Guardar log CSV
    df = pd.DataFrame(metrics_log)
    csv_path = os.path.join(output_dir, f"{model_name}_training_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"Log de métricas guardado: {csv_path}")

    return model
