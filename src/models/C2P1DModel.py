import torch.nn as nn
import torchvision

def create_c2p1d_model(num_classes):
    # model = torchvision.models.video.r3d_18(pretrained=True)
    model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
    model.fc = nn.Linear(512, 2, bias=False)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model