import optuna
import torch
from torch.utils.data import DataLoader
from models.ModelFactory import ModelFactory
from ds.DatasetFactory import DatasetFactory
from ds.TransformFactory import TransformFactory
from train.Train import train_model
from train.Eval import evaluate_model
from datetime import datetime
import pandas as pd
import os

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [96])

    model_name = "resnet50"
    transformer_name, tfms = TransformFactory.get_transform_by_name("trans_res256bilinear_crop224")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 2
    input_dir = "/mnt/d2/competicion"

    train_ds = DatasetFactory.create(
        num_classes=num_classes,
        protocol_file=f"{input_dir}/Protocol-train.txt",
        root_dir=f"{input_dir}/Data-train",
        transform=tfms,
        is3d=False
    )
    eval_ds = DatasetFactory.create(
        num_classes=num_classes,
        protocol_file=f"{input_dir}/Protocol-val.txt",
        root_dir=f"{input_dir}/Data-val",
        transform=tfms,
        is3d=False
    )
    eval2_ds = DatasetFactory.create(
        num_classes=num_classes,
        protocol_file=f"{input_dir}/Protocol-vt.txt",
        root_dir=f"{input_dir}/Data-vt",
        transform=tfms,
        is3d=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=8)
    eval2_loader = DataLoader(eval2_ds, batch_size=8)

    model = ModelFactory.create(model_name, num_classes=num_classes, device=device, pretrained=True, multiGPU=False)

    model = train_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        eval2_loader=eval2_loader,
        transformer_name=transformer_name,
        tfms=tfms,
        model_name=f"{model_name}_optuna",
        num_classes=num_classes,
        class_weights_path=f"class_weights_{num_classes}clases.npy",
        checkpoint_every=1000,
        epochs=5,
        device=device,
        input_dir=input_dir,
        output_dir="optuna_logs",
        hyperparams={"lr": lr, "weight_decay": weight_decay}
    )

    metrics = evaluate_model(model, eval_loader, device, num_classes, desc="eval")
    auc = metrics.get("AUC", 0.0)
    trial.set_user_attr("EER", metrics.get("EER", None))  # opcional para mostrar EER también
    return auc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    # Imprimir mejores hiperparámetros
    print("✅ Mejores hiperparámetros encontrados:")
    print(study.best_params)
    print("\n📈 Mejores resultados (ordenados por AUC):")

    # Guardar y mostrar los trials como tabla
    records = []
    for t in study.trials:
        records.append({
            "Trial": t.number,
            "AUC": t.value,
            "LR": t.params.get("lr"),
            "Weight Decay": t.params.get("weight_decay"),
            "Batch Size": t.params.get("batch_size"),
            "EER": t.user_attrs.get("EER")
        })

    df = pd.DataFrame(records)
    df_sorted = df.sort_values(by="AUC", ascending=False)
    print(df_sorted.to_string(index=False))

    # Exportar como CSV para el paper
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = f"optuna_results_{timestamp}.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n📁 Resultados exportados a: {csv_path}")