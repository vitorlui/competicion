import argparse
from train.Train import train_model
from ds.TransformFactory import TransformFactory
from models.ModelFactory import ModelFactory
from ds.DatasetFactory import DatasetFactory
from torch.utils.data import DataLoader

# Lista de modelos 3D
models_3d = ["r2plus1d_18", "mc3_18", "r3d_18"]

def is_3d_model(model_name):
    models_3d = ["r2plus1d_18", "mc3_18", "r3d_18"]
    return model_name in models_3d

def main():
    parser = argparse.ArgumentParser(description="Train a model for FAS classification")
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g. vit_b_32, resnet50)')
    parser.add_argument('--transformer', type=str, required=True, help='Transformer name from TransformFactory')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device (default: cuda:0)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--every_epoch', type=int, default=5, help='Checkpoint frequency (default: 5)')
    parser.add_argument('--classes', type=int, default=2, help='Number of output classes (2 or 3)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--preproc', type=bool, default=False)
    parser.add_argument('--multiGPU', type=bool, default=False)
    

    args = parser.parse_args()

    transform_name = args.transformer
    transform_name, tfms = TransformFactory.get_transform_by_name(transform_name)
    num_classes_iter = args.classes

    print("Usando transformación:", transform_name)

    if not args.preproc:
        train_ds = DatasetFactory.create(
            num_classes=num_classes_iter,
            protocol_file="Protocol-train.txt",
            root_dir="Data-train",
            transform=tfms,
            is3d= is_3d_model(args.model)
        )
    else:
        print("Usando preproc")
        if "bicubic" in transform_name:
            print("Usando preproc bicubic")
            protocol_file_preproc = "Protocol-trainPreProcBicubic.txt"
            root_dir_preproc = "Data-trainPrepProcResize256bicubic_crop224"
        else:
            print("Usando preproc bilinear")
            protocol_file_preproc = "Protocol-trainPreProcBilinear.txt"
            root_dir_preproc = "Data-trainPrepProcResize256bilinear_crop224"
        
        train_ds = DatasetFactory.create(
            num_classes=num_classes_iter,
            protocol_file=protocol_file_preproc,
            root_dir=root_dir_preproc, 
            transform=TransformFactory.get_postprocessing_transform()
        )


    eval_ds = DatasetFactory.create(
        num_classes=num_classes_iter,
        protocol_file="Protocol-val.txt",
        root_dir="Data-val",
        transform=tfms,
        is3d= is_3d_model(args.model)
    )
    eval2_ds = DatasetFactory.create(
        num_classes=num_classes_iter,
        protocol_file="Protocol-vt.txt",
        root_dir="Data-vt",
        transform=tfms,
        is3d= is_3d_model(args.model)
    )

    model = ModelFactory.create(args.model, num_classes=num_classes_iter, device=args.device, pretrained=args.pretrained, multiGPU=args.multiGPU)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=16)
    eval2_loader = DataLoader(eval2_ds, batch_size=16)

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
        output_dir=f"ck_{args.model}_{num_classes_iter}_pt{1 if args.pretrained else 0}_{transform_name}"
    )

if __name__ == "__main__":
    main()
