import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.linalg as la
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from robustbench import load_model
from energyUtils import rand_weights, compute_energy
from PIL import Image
from CustomWRN.wrn_cifar import wrn_28_10, WideResNet

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = CIFAR10('./data', train=False, download=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

x, y = next(iter(dataloader))
print(x.shape)

models = {
        '9-9-5_10-10-10_ReLU' : [nn.ReLU, [9, 9, 5], [10, 10, 10]],
        '5-1-7_10-10-10_ReLU' : [nn.ReLU, [5, 1, 7], [10, 10, 10]],
        '1-3-7_10-10-10_ReLU' : [nn.ReLU, [1, 3, 7], [10, 10, 10]],
        '1-1-7_10-10-10_ReLU' : [nn.ReLU, [1, 1, 7], [10, 10, 10]]
    }

models = {
        '7-9-3_10-10-10_ReLU' : [nn.ReLU, [7, 9, 3], [10, 10, 10]],
        '5-1-5_10-10-10_ReLU' : [nn.ReLU, [5, 1, 5], [10, 10, 10]],
        '3-1-5_10-10-10_ReLU' : [nn.ReLU, [3, 1, 5], [10, 10, 10]],
        '1-1-5_10-10-10_ReLU' : [nn.ReLU, [1, 1, 5], [10, 10, 10]]
    }

for model_name in models:
    model = WideResNet(nn.Conv2d, nn.Linear, act_fn=models[model_name][0], custom_depths=models[model_name][1], custom_widen_factor=models[model_name][2], dropRate=0)
    model.eval()
    rand_weights(model)
    print(torch.max(la.vector_norm(model(x), ord=float('inf'), dim=(-1))))

# Sanity check
v = torch.tensor([1., 1., 1.])
A = torch.diag(v)
print(la.matrix_norm(A, ord=float('inf')))