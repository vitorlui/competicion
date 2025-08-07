import optuna
import matplotlib.pyplot as plt
import argparse
import os

# CLI arguments
parser = argparse.ArgumentParser(description="Plot AUC and EER from an Optuna study")
parser.add_argument('--model', type=str, required=True, help='Model name (e.g., vit_b_16)')
parser.add_argument('--pretrained', type=int, default=1, help='Use pretrained weights (1 or 0)')
parser.add_argument('--db-path', type=str, default=None, help='Optional path to Optuna DB (auto-inferred if not set)')
args = parser.parse_args()

model_name = args.model
pretrained = args.pretrained
study_name = f"{model_name}_optuna_pt{pretrained}"

# Infer DB path if not provided
if args.db_path:
    db_path = args.db_path
else:
    db_path = f"sqlite:///optuna_{study_name}_study.db"

# Load study
study = optuna.load_study(study_name=study_name, storage=db_path)

# Extract trial data
trials = [t for t in study.trials if t.value is not None]
trial_numbers = [t.number for t in trials]
aucs = [t.value for t in trials]
eers = [t.user_attrs.get("EER", None) for t in trials]

# Filter valid EERs
trial_numbers_eer = [t for t, e in zip(trial_numbers, eers) if e is not None]
eers = [e for e in eers if e is not None]

# Plot AUC
plt.figure(figsize=(10, 4))
plt.plot(trial_numbers, aucs, marker='o', label="AUC")
plt.title(f"AUC per Trial - {model_name}")
plt.xlabel("Trial Number")
plt.ylabel("AUC Score")
plt.grid(True)
plt.tight_layout()
auc_path = f"optuna_{model_name}_pt{pretrained}_auc.png"
plt.savefig(auc_path)
print(f"AUC plot saved to: {auc_path}")

# Plot EER
plt.figure(figsize=(10, 4))
plt.plot(trial_numbers_eer, eers, marker='o', color='red', label="EER")
plt.title(f"EER per Trial - {model_name}")
plt.xlabel("Trial Number")
plt.ylabel("EER Score")
plt.grid(True)
plt.tight_layout()
eer_path = f"optuna_{model_name}_pt{pretrained}_eer.png"
plt.savefig(eer_path)
print(f"EER plot saved to: {eer_path}")