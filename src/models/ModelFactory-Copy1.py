import torch
import torch.nn as nn
import torchvision.models as models

class ModelFactory:
    """Factory class to create models with adjusted final layers for given num_classes."""
    
    @staticmethod
    def resnet50(num_classes: int, device: torch.device):
        model = models.resnet50(pretrained=True)  # Carga modelo ResNet-50 preentrenado
        # Reemplaza la capa final fully-connected por una nueva con num_classes salidas
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)
    
    @staticmethod
    def resnet152(num_classes: int, device: torch.device):
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)
    
    @staticmethod
    def resnext50_32x4d(num_classes: int, device: torch.device):
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)
    
    @staticmethod
    def resnext101_64x4d(num_classes: int, device: torch.device):
        model = models.resnext101_64x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)
    
    @staticmethod
    def vit_b_16(num_classes: int, device: torch.device):
        model = models.vit_b_16(pretrained=True)
        # Reemplaza la cabeza de clasificación del ViT
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model.to(device)
    
    @staticmethod
    def vit_b_32(num_classes: int, device: torch.device):
        model = models.vit_b_32(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model.to(device)
