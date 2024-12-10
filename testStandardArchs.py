import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from energyUtils import test_pgd_impact

os.makedirs(os.path.join(os.getcwd(), 'energyTesting/means/Resnets'), exist_ok=True)

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, trust_repo=True)
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = CIFAR10('./data', train=False, download=False, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

test_pgd_impact([1, 2, 5], model, 'Resnet56', dataloader)


