import os
from torchvision import transforms
from torch.utils.data import DataLoader
from ds.FilteredDataset import FilteredDataset

def main():
    protocol_path = "Protocol-train.txt"
    root_dir = "Data-train"  # Asegúrate que es la ruta correcta

    # Transformaciones de red (resize, crop, normalización)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4603, 0.3696, 0.3388],
            std=[0.2246, 0.2059, 0.1994]
        )
    ])

    # Dataset con filtros activados
    dataset = FilteredDataset(
        protocol_file=protocol_path,
        root_dir=root_dir,
        apply_filters=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for video_tensor, label in loader:
        print(f"📦 Video tensor shape: {video_tensor.shape}")  # (B, F, C, H, W)
        print(f"🏷️ Labels: {label}")
        break


if __name__ == "__main__":
    main()
