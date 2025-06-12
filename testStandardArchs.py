import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from time import time
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from energyUtils import test_pgd_impact, rand_weights, git_update, fit_image
from robustbench.utils import load_model
from CustomWRN.wrn_cifar import wrn_28_10, WideResNet

os.makedirs(os.path.join(os.getcwd(), 'means'), exist_ok=True)
os.environ['TORCH_HOME'] = 'E:\\torch\\cache\\checkpoints'


print('Computing on cuda? ', torch.cuda.is_available())
device = ('cuda' if torch.cuda.is_available() else 'cpu')

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = CIFAR10('./data', train=False, download=True, transform=preprocess)
x, y_i = dataset[0]
y = torch.tensor(y_i)
x = x.unsqueeze(0)
y = y.unsqueeze(0)
x = x.to(device)
y = y.to(device)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

models = {
        '9-9-5_10-10-10_ReLU' : [nn.ReLU, [9, 9, 5], [10, 10, 10]],
        '5-1-7_10-10-10_ReLU' : [nn.ReLU, [5, 1, 7], [10, 10, 10]],
        '1-3-7_10-10-10_ReLU' : [nn.ReLU, [1, 3, 7], [10, 10, 10]],
        '1-1-7_10-10-10_ReLU' : [nn.ReLU, [1, 1, 7], [10, 10, 10]]
    }

for i in range(10):
    torch.manual_seed(i)

    start = time()

    for model_name in models:
        #model_name = f"Resnet{model_numb}"
        #model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_resnet{model_numb}", pretrained=True, trust_repo=True)
        #model = load_model(model_name, dataset='cifar10', threat_model='Linf')
        model = WideResNet(nn.Conv2d, nn.Linear, act_fn=models[model_name][0], custom_depths=models[model_name][1], custom_widen_factor=models[model_name][2], dropRate=0)
        model.train()
        model.to(device)
        rand_weights(model, False, False)
        print(f'Testing {model_name}')
        #fit_image(model, x, y)
        test_pgd_impact([1, 2, 5, 10, 20], model, model_name, dataloader, f'RandCustom/RandDepths55M/CustomWRNRandSameParams{i}_40M', device=device, alpha=2/255, eps=8/255)
        del model
    end = time()

    print(f"Test suite took {(end - start) / 60 : .2f} minutes")
    print(git_update())