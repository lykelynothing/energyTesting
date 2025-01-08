import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from energyUtils import test_pgd_impact, rand_weights
from robustbench.utils import load_model

os.makedirs(os.path.join(os.getcwd(), 'means'), exist_ok=True)
os.environ['TORCH_HOME'] = 'E:\\torch\\cache\\checkpoints'


print('Computing on cuda? ', torch.cuda.is_available())
device = ('cuda' if torch.cuda.is_available() else 'cpu')

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = CIFAR10('./data', train=False, download=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

# Resnets
model_numbers = ['20', '32', '44', '56']

# WRN 28-10
models = {'Standard' : '0.0%', 'Sehwag2020Hydra' : '57.14%', 'Wang2020Improving' : '56.29%', 'Zhang2020Geometry' : '59.64%',
              'Gowal2020Uncovering_28_10_extra' : '62.76%', 'Wang2023Better_WRN-28-10' : '67.31%', 'Cui2023Decoupled_WRN-28-10' : '67.74%'}

'''
# WRN 34-10
models = {'Zhang2020Attacks' : '53.51%', 'Zhang2019You' : '44.83%', 'Wu2020Adversarial' : '56.17%',
            'Chen2024Data_WRN_34_10' : '57.30%', 'Addepalli2021Towards_WRN34' : '58.04%', 'Sehwag2021Proxy' : '60.27%',
            'Rade2021Helper_extra' : '62.83%', 'Huang2020Self' : '53.34%', 'Sitawarin2020Improving' : '50.72%', 
            'Chen2020Efficient' : '51.12%', 'Cui2020Learnable_34_10' : '52.86%', 'Chen2021LTD_WRN34_10' : '56.94%'}


#models = {'Sitawarin2020Improving' : '50.72%', 'Chen2020Efficient' : '51.12%',
           'Cui2020Learnable_34_10' : '52.86%', 'Chen2021LTD_WRN34_10' : '56.94%'}

#models = {'Chen2020Efficient' : '51.12%', 'Cui2020Learnable_34_10' : '52.86%'}
'''

models = {'Gowal2020Uncovering_28_10_extra' : '62.76%', 'Wang2023Better_WRN-28-10' : '67.31%', 'Cui2023Decoupled_WRN-28-10' : '67.74%'}

for model_name in models:
    #model_name = f"Resnet{model_numb}"
    #model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_resnet{model_numb}", pretrained=True, trust_repo=True)
    
    model = load_model(model_name, dataset='cifar10', threat_model='Linf')
    model.eval()
    model.to(device)
    #rand_weights(model)
    print(f'Testing {model_name}')
    test_pgd_impact([20, 50], model, model_name, dataloader, 'WRN28-10', model_rob=models[model_name], device=device)
    del model
