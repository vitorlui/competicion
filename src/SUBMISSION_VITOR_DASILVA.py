import argparse
from train.Train import train_iteractive
from ds.TransformFactory import TransformFactory
from models.ModelFactory import ModelFactory
from ds.DatasetFactory import DatasetFactory
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description="Train a model for FAS classification")
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g. vit_b_32, resnet50)')
    parser.add_argument('--transformer', type=str, required=True, help='Transformer name from TransformFactory')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device (default: cuda:0)')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--every_epoch', type=int, default=15, help='Checkpoint frequency (default: 5)')
    parser.add_argument('--classes', type=int, default=2, help='Number of output classes (2 or 3)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--preproc', type=bool, default=False)
    parser.add_argument('--multiGPU', type=bool, default=False)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--input_dir', type=str, default="/mnt/d2/competicion")
    parser.add_argument('--label', type=str, default=None)
    
    label = "iteract"
    args = parser.parse_args()
    # print(f"args:{args}")
    transform_name = args.transformer
    num_classes_iter = args.classes

    tfms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean =  [0.4603, 0.3696, 0.3388]
                                                         , std = [0.2246, 0.2059, 0.1994]
                                   )])

    print("Usando transformación:", transform_name)    

    # Diccionario de etiquetas legibles
    label_map = {
        "0_0_0": "Live Face",
        "1_0_0": "Print (2D Attack)",
        "1_0_1": "Replay (2D Attack)",
        "1_0_2": "Cutouts (2D Attack)",
        "1_1_0": "Transparent (3D Attack)",
        "1_1_1": "Plaster (3D Attack)",
        "1_1_2": "Resin (3D Attack)",
        "2_0_0": "Attribute-Edit (Digital Manipulation)",
        "2_0_1": "Face-Swap (Digital Manipulation)",
        "2_0_2": "Video-Driven (Digital Manipulation)",
        "2_1_0": "Pixel-Level (Digital Adversarial)",
        "2_1_1": "Semantic-Level (Digital Adversarial)",
        "2_2_0": "ID_Consistent (Digital Generation)",
        "2_2_1": "Style (Digital Generation)",
        "2_2_2": "Prompt (Digital Generation)"
    }


    # Distribución real por clase
    class_distribution = {
        "2_1_0": 8364,
        "2_0_1": 6160,
        "2_1_1": 3757,
        "2_0_2": 1540,
        "2_0_0": 1476,
        "0_0_0": 839,
        "1_0_1": 109,
        "1_0_2": 79,
        "1_0_0": 43,
        # clases con 0 no se incluyen
    }

    # Ordenar clases de menos a más representadas, excluyendo la bonafide "0_0_0"
    ordered_fraud_labels = OrderedDict(
        sorted(
            {k: v for k, v in class_distribution.items() if k != "0_0_0"}.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    # Lista para guardar los datasets
    datasets_iterativos = []

    # Bonafide fijo
    bonafide_label = "0_0_0"
    bonafide_prefixes = [bonafide_label.split("_")[0] + "_"]

    # Iteraciones
    for i, (fraud_label, _) in enumerate(ordered_fraud_labels.items()):
        fraud_prefixes = [fraud_label.split("_")[0] + "_"]

        print(f"[Iteración {i+1}] Bonafide: {label_map[bonafide_label]} ({bonafide_prefixes})")
        print(f"Fraude: {label_map[fraud_label]} ({fraud_prefixes})")

        train_ds = DatasetFactory.create(
            num_classes=21,
            protocol_file=f"{args.input_dir}/Protocol-train.txt",
            root_dir=f"{args.input_dir}/Data-train",
            transform=tfms,
            is3d=False,
            bonafide_prefixes=bonafide_prefixes,
            fraud_prefixes=fraud_prefixes
        )

        datasets_iterativos.append((fraud_label, train_ds))
        # break
    
    print("Size", len(datasets_iterativos))

    train_ds_all = DatasetFactory.create(
            num_classes=2,
            protocol_file=f"{args.input_dir}/Protocol-train.txt",
            root_dir=f"{args.input_dir}/Data-train",
            transform=tfms,
            is3d=False,
            bonafide_prefixes=bonafide_prefixes,
            fraud_prefixes=fraud_prefixes
        )

    datasets_iterativos.append(("all", train_ds_all))
    
    print("Size with all", len(datasets_iterativos))

    eval_ds = DatasetFactory.create(
        num_classes=2,
        protocol_file=f"{args.input_dir}/Protocol-val.txt",
        root_dir=f"{args.input_dir}/Data-val",
        transform=tfms,
        is3d= False
    )

    eval2_ds = DatasetFactory.create(
        num_classes=2,
        protocol_file=f"{args.input_dir}/Protocol-vt.txt",
        root_dir=f"{args.input_dir}/Data-vt",
        transform=tfms,
        is3d= False
    )

    model = ModelFactory.create(args.model, num_classes=num_classes_iter, device=args.device, pretrained=args.pretrained, multiGPU=args.multiGPU)

    train_dataloaders = [
                            DataLoader(
                                dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2,  # ajusta según tu máquina
                            )
                            for _, dataset in datasets_iterativos
                        ]
    
    eval_loader = DataLoader(eval_ds, batch_size=16)
    eval2_loader = DataLoader(eval2_ds, batch_size=16)

    model = train_iteractive(
        model=model,
        train_loaders=train_dataloaders, 
        eval_loader=eval_loader,
        eval2_loader=eval2_loader,
        transformer_name=transform_name,
        tfms=tfms,
        model_name=f"{args.model}_{num_classes_iter}c",
        num_classes=num_classes_iter,
        class_weights_path=f"class_weights_{num_classes_iter}clases.npy",
        checkpoint_every=args.every_epoch,
        epochs=args.epochs,
        device=args.device,
        input_dir=args.input_dir,
        output_dir=f"ck_{args.model}_{num_classes_iter}_pt{1 if args.pretrained else 0}_{transform_name}{label}"
    )


if __name__ == "__main__":
    main()