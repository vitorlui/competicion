import argparse
from time import sleep
from train.Train import train_model
from ds.TransformFactory import TransformFactory
from models.ModelFactory import ModelFactory
from ds.DatasetFactory import DatasetFactory
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import ConcatDataset, Subset
from copy import deepcopy
from collections import Counter
import torch

def print_class_distribution(dataset, name="Dataset"):
    """
    Imprime la distribución de clases de un Dataset o ConcatDataset.

    Args:
        dataset: Un Dataset con muestras del tipo (image, label).
        name (str): Nombre para identificar el dataset en el print.
    """
    labels = []
    
    # Soporte para ConcatDataset o Dataset normal
    if hasattr(dataset, 'datasets'):  # ConcatDataset
        for d in dataset.datasets:
            labels.extend([label for _, label in d])
    else:
        labels = [label for _, label in dataset]

    counter = Counter(labels)
    print(f"📊 Distribución de clases en {name}: {dict(counter)}")
    
def is_3d_model(model_name):
    models_3d = ["r2plus1d_18", "mc3_18", "r3d_18"
                 , "r2plus1d_18_2c", "mc3_18_2c", "r3d_18_2c"
                 , "r2plus1d_18_3c", "mc3_18_3c", "r3d_18_3c"
                 , "r2plus1d_18_6c", "mc3_18_6c", "r3d_18_6c"
                 , "r2plus1d_18_15c", "mc3_18_15c", "r3d_18_15c"
                 ]
    return model_name in models_3d



def balance_bonafide(bonafide_ds, target_size):
    n = len(bonafide_ds)
    factor = target_size // n
    remainder = target_size % n

    datasets = [bonafide_ds] * factor
    if remainder > 0:
        datasets.append(Subset(bonafide_ds, list(range(remainder))))
    return ConcatDataset(datasets)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():
    parser = argparse.ArgumentParser(description="Train a model for FAS classification")
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g. vit_b_32, resnet50)')
    parser.add_argument('--transformer', type=str, required=True, help='Transformer name from TransformFactory')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device (default: cuda:0)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--every_epoch', type=int, default=5, help='Checkpoint frequency (default: 5)')
    parser.add_argument('--classes', type=int, default=2, help='Number of output classes (2 or 3)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--preproc', type=str2bool, default=False)
    parser.add_argument('--multiGPU', type=str2bool, default=False)
    parser.add_argument('--test_mode', type=str2bool, default=False)
    parser.add_argument('--input_dir', type=str, default="/mnt/d2/competicion")
    parser.add_argument('--label', type=str, default=None)
    

    args = parser.parse_args()
    print(f"args:{args}")
    transform_name = args.transformer
    transform_name, tfms = TransformFactory.get_transform_by_name(transform_name)
    num_classes_iter = args.classes

    # tfms = transforms.Compose([transforms.Resize([112, 112]),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean = [0.43216, 0.394666, 0.37645]
    #                                                      , std = [0.22803, 0.22145, 0.216989]
    #                                )])
    
    # tfms = transforms.Compose([transforms.Resize([256, 256]),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean =  [0.4603, 0.3696, 0.3388]
    #                                                      , std = [0.2246, 0.2059, 0.1994]
    #                                )])
    
    # transform_name = "videoTfms"  

    print("Usando transformación:", transform_name)
    print("is_3d_model:", is_3d_model(args.model))
    
    if not args.preproc:
        train_ds = DatasetFactory.create(
            num_classes=num_classes_iter,
            protocol_file=f"{args.input_dir}/Protocol-train.txt",
            root_dir=f"{args.input_dir}/Data-train",
            transform=tfms,
            is3d= is_3d_model(args.model)
        )

        print_class_distribution(train_ds, name="Antes del balanceo")
      

        # bonafide_samples = [s for s in train_ds.samples if s[1] == 0]
        # fraud_samples = [s for s in train_ds.samples if s[1] >= 1]

        # bonafide_ds = deepcopy(train_ds)
        # bonafide_ds.samples = bonafide_samples

        # fraud_ds = deepcopy(train_ds)
        # fraud_ds.samples = fraud_samples

        # balanced_bonafide_ds = balance_bonafide(bonafide_ds, len(fraud_ds))

        # train_ds = ConcatDataset([balanced_bonafide_ds, fraud_ds])
        
        # print_class_distribution(train_ds, name="Después del balanceo")

        if args.test_mode:
            print("Test mode ON")
            # Teste mode
            train_ds = DatasetFactory.create(
                num_classes=num_classes_iter,
                protocol_file=f"{args.input_dir}/Protocol-vt.txt",
                root_dir=f"{args.input_dir}/Data-vt",
                transform=tfms,
                is3d= is_3d_model(args.model)
            )
    else:
        print("Usando preproc")
        if is_3d_model(args.model):
            print("Usando preproc 3D")
            protocol_file_preproc = "Protocol-train.txt"
            root_dir_preproc = f"{args.input_dir}/Data-train3D"

        elif "bicubic" in transform_name:
            print("Usando preproc bicubic")
            protocol_file_preproc = "Protocol-train_bicubic.txt"
            root_dir_preproc = f"{args.input_dir}/Data-train_bicubic"
        else:
            print("Usando preproc bilinear")
            protocol_file_preproc = "Protocol-train_bilinear.txt"
            root_dir_preproc = f"{args.input_dir}/Data-train_bilinear"
        
        train_ds = DatasetFactory.create(
            num_classes=num_classes_iter,
            protocol_file=protocol_file_preproc,
            root_dir=root_dir_preproc, 
            transform=TransformFactory.get_postprocessing_transform(),
            is3d= is_3d_model(args.model)
        )

    eval_ds = DatasetFactory.create(
        num_classes=num_classes_iter,
        protocol_file=f"{args.input_dir}/Protocol-val.txt",
        root_dir=f"{args.input_dir}/Data-val",
        transform=tfms,
        is3d= is_3d_model(args.model)
    )

    if args.test_mode:
        # Teste mode
        eval_ds = DatasetFactory.create(
                num_classes=num_classes_iter,
                protocol_file=f"{args.input_dir}/Protocol-vt.txt",
                root_dir=f"{args.input_dir}/Data-vt",
                transform=tfms,
                is3d= is_3d_model(args.model)
            )
    
    eval2_ds = DatasetFactory.create(
        num_classes=num_classes_iter,
        protocol_file=f"{args.input_dir}/Protocol-vt.txt",
        root_dir=f"{args.input_dir}/Data-vt",
        transform=tfms,
        is3d= is_3d_model(args.model)
    )

    model = ModelFactory.create(args.model, num_classes=num_classes_iter, device=args.device, pretrained=args.pretrained, multiGPU=args.multiGPU)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=8)
    eval2_loader = DataLoader(eval2_ds, batch_size=8)
    label = f"_{args.label}" if args.label else ""
    
    model = train_model(
        model=model,
        train_loader=train_loader,
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
