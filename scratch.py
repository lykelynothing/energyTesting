import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from robustbench import load_model
from energyUtils import rand_weights, compute_energy
from PIL import Image
from CustomWRN.wrn_cifar import wrn_28_10, WideResNet


models = {
        '9-9-5_10-10-10_ReLU' : [nn.ReLU, [9, 9, 5], [10, 10, 10]],
        '5-1-7_10-10-10_ReLU' : [nn.ReLU, [5, 1, 7], [10, 10, 10]],
        '1-3-7_10-10-10_ReLU' : [nn.ReLU, [1, 3, 7], [10, 10, 10]],
        '1-1-7_10-10-10_ReLU' : [nn.ReLU, [1, 1, 7], [10, 10, 10]]
    }

model_name = '9-9-5_10-10-10_ReLU'
model = WideResNet(nn.Conv2d, nn.Linear, act_fn=models[model_name][0], custom_depths=models[model_name][1], custom_widen_factor=models[model_name][2], dropRate=0)
for n, module in model.named_modules():
    if 'conv' in n.lower():
        print(module.weight.data.shape)