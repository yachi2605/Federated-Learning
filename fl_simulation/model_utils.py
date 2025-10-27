import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name: str, num_classes: int):
    """Get and adapt ResNet-34 for CIFAR-100"""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def evaluate(model, test_loader, device):
    """Test model on CIFAR-100"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return test_loss / len(test_loader), 100. * correct / total