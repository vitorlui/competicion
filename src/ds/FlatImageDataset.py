from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

class FlatImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform if transform else transforms.ToTensor()
        self.image_paths = sorted([p for p in self.folder_path.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), 0  # label dummy
