import torch
import torch.nn as nn
from robustbench import load_model
from CustomWRN.wrn_cifar import wrn_28_10
from energyUtils import rand_weights
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

# Before going into conv1 of first block of second stage, something happens in std (probably shortcut)

x = torch.rand(3, 32, 32)

outputs = {}

def hook_fn(module, input, output):
    outputs[module] = input[0], output[0]

model = wrn_28_10(nn.Conv2d, nn.Linear, nn.SiLU, 'kaiming_normal', dropRate=0)
rand_weights(model)

for block in list(model.children())[1:-3]:
   block.layer[0].conv1.register_forward_hook(hook_fn)

# Print mean of outputs and of inner layers outputs
print('Custom outputs: ', torch.mean(model((torch.unsqueeze(x, 0)))).item())

print('Custom inner: ', [torch.mean(outputs[t][1]).item() for t in outputs.keys()])

print('Mean conv weight: ', torch.mean(model.block2.layer[2].conv1.weight).item())


# Same for standard
outputs = {}

model = load_model('Standard', dataset='cifar10', threat_model='Linf')
rand_weights(model)

for block in list(model.children())[1:-3]: # Ignores first and last conv and bn
    block.layer[0].conv1.register_forward_hook(hook_fn)

print('Standard: ', torch.mean(model((torch.unsqueeze(x, 0)))).item())

print('Inner outputs: ', [torch.mean(outputs[t][1]).item() for t in outputs.keys()])

print('Conv weight: ', torch.mean(model.block2.layer[2].conv1.weight).item())