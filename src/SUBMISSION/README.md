# GCD-UDL Submission - Face Anti-Spoofing Challenge 2025

This repository contains the official submission by the **GCD-UDL** team for the 6th Face Anti-Spoofing Challenge 2025.

## 📁 Directory Structure

```
.
├── class_weights_2clases.npy
├── dataset
│   ├── Data-test
│   ├── Data-train
│   ├── Data-val
│   ├── Protocol-test.txt
│   ├── Protocol-train.txt
│   ├── Protocol-val-notlabeled.txt -> For output txt generation
│   └── Protocol-val.txt -> for evaluation
├── EXEC_SUBMISSION_GCD_UDL_VITOR_DASILVA.sh
├── outputs
│   ├── ck_vit_b_16_2_pt1_trans_res256bilinear_nocropiteract_0207_1053
│   ├── ck_vit_b_16_2_pt1_trans_res256bilinear_nocropiteract_0207_1058
│   └── ck_vit_b_16_2_pt1_trans_res256bilinear_nocropiteract_0207_1059
├── README.md
└── SUBMISSION_GCD_UDL_VITOR_DASILVA.py
```

## ⚙️ Requirements

This code was tested with the following environment:

* Python 3.8+
* PyTorch ≥ 2
* torchvision
* scikit-learn
* numpy
* Pillow (PIL)
* tqdm

## ▶️ How to Run

To reproduce the full process (training and evaluation), simply run the following bash script:

```bash
./EXEC_SUBMISSION_GCD_UDL_VITOR_DASILVA.sh
```

This script internally executes the main Python script with the appropriate arguments.

### Arguments passed:

* `--model vit_b_16`
* `--transformer trans_res256bilinear_nocrop`
* `--device cuda:0`
* `--epochs 15`
* `--every_epoch 15`
* `--classes 2`
* `--batch_size 64`
* `--input_dir ./dataset`

## 💾 Output

All outputs including checkpoints and metric logs will be stored in the `outputs/` folder.

## ℹ️ Notes

* For full training, change `--epochs` to 15 in `EXEC_SUBMISSION_GCD_UDL_VITOR_DASILVA.sh`.
* The `dataset/` folder must include all required image folders (`Data-train`, `Data-val`, `Data-test`) and the corresponding protocol files. You can change the input_dir argmument to point to your dataset directory.

## 📬 Contact

If you have any questions, please contact:

**Vítor Luiz da Silva Verbel**
Email: [vitor.dasilva@udl.cat](mailto:vitor.dasilva@udl.cat)
Affiliation: Department of Computer Engineering and Digital Design, University of Lleida