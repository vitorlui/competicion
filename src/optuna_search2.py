import argparse
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

# Argumentos desde línea de comandos
parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
parser.add_argument('--model', type=str, required=True, help='Model name (e.g., vit_b_16, resnet50)')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--pretrained', type=str2bool, default=True, help='Use pretrained weights (True/False)')


# parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights (True/False)')
parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs per trial')
parser.add_argument('--cuda', type=int, default=1, help='Cuda')
args = parser.parse_args()

# Variables globales
CHECKPOINT_DIR = "optuna_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_NAME = args.model
PRETRAINED = args.pretrained
NUM_TRIALS = args.trials
EPOCHS = args.epochs

STUDY_NAME = f"{MODEL_NAME}_optuna_pt{int(PRETRAINED)}"
STORAGE_PATH = f"sqlite:///optuna_{STUDY_NAME}_study.db"

NUM_CLASSES = 2
INPUT_DIR = "/mnt/d2/competicion"
TRANSFORM_NAME = "trans_res256bilinear_crop224"

# Transform y dispositivo
transformer_name, tfms = TransformFactory.get_transform_by_name(TRANSFORM_NAME)
device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

def get_dataloaders(batch_size):
    train_ds = DatasetFactory.create(
        num_classes=NUM_CLASSES,
        protocol_file=f"{INPUT_DIR}/Protocol-train.txt",
        root_dir=f"{INPUT_DIR}/Data-train",
        transform=tfms,
        is3d=False
    )
    eval_ds = DatasetFactory.create(
        num_classes=NUM_CLASSES,
        protocol_file=f"{INPUT_DIR}/Protocol-val.txt",
        root_dir=f"{INPUT_DIR}/Data-val",
        transform=tfms,
        is3d=False
    )
    eval2_ds = DatasetFactory.create(
        num_classes=NUM_CLASSES,
        protocol_file=f"{INPUT_DIR}/Protocol-vt.txt",
        root_dir=f"{INPUT_DIR}/Data-vt",
        transform=tfms,
        is3d=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=8)
    eval2_loader = DataLoader(eval2_ds, batch_size=8)

    return train_loader, eval_loader, eval2_loader

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64])

    train_loader, eval_loader, eval2_loader = get_dataloaders(batch_size)

    model = ModelFactory.create(MODEL_NAME, num_classes=NUM_CLASSES, device=device, pretrained=bool(PRETRAINED), multiGPU=False)

    model, metrics_log = train_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        eval2_loader=eval2_loader,
        transformer_name=transformer_name,
        tfms=tfms,
        model_name=f"{MODEL_NAME}_optuna",
        num_classes=NUM_CLASSES,
        class_weights_path=f"class_weights_{NUM_CLASSES}clases.npy",
        checkpoint_every=1000,
        epochs=EPOCHS,
        device=device,
        input_dir=INPUT_DIR,
        output_dir="optuna_logs",
        hyperparams={"lr": lr, "weight_decay": weight_decay}
    )

    metrics = evaluate_model(model, eval_loader, device, NUM_CLASSES, desc="eval")
    auc = metrics.get("AUC", 0.0)
    eer = metrics.get("EER", None)
    trial.set_user_attr("EER", eer)

    

    # Guardar solo el mejor modelo
    if not trial.study.best_trial or auc > trial.study.best_value:
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Nuevo mejor modelo guardado en: {best_model_path} (AUC={auc:.4f})")

    return auc

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=NUM_TRIALS)

    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)

    print("\n Mejores resultados (ordenados por AUC):")
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = f"optuna_results_{MODEL_NAME}_{timestamp}.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n Resultados exportados a: {csv_path}")