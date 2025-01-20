import torch
import torch.nn as nn
from robustbench import load_model
from CustomWRN.wrn_cifar import wrn_28_10, wrn_34_10
from energyUtils import rand_weights, compute_energy
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

# Before going into conv1 of first block of second stage, something happens in std (probably shortcut)

x = torch.rand(3, 32, 32)

outputs = {}

def hook_fn(module, input, output):
    outputs[module] = input[0], output[0]


model = wrn_28_10(nn.Conv2d, nn.Linear, nn.SiLU, 'kaiming_normal', dropRate=0)
model.train()
rand_weights(model, inplace=False, track=False)
# check out second block first stage
for block in list(model.children())[1:-3]:
   block.layer[3].conv2.register_forward_hook(hook_fn)

# Print mean of outputs and of inner layers outputs
print('Custom outputs: ', compute_energy(model((torch.unsqueeze(x, 0)))).item())

print('Inner inputs 4th block first stage: ', [torch.mean(outputs[t][0]).item() for t in outputs.keys()])

# Same for standard
outputs = {}

model = load_model('Standard', dataset='cifar10', threat_model='Linf')
model.train()
rand_weights(model, inplace=False, track=False)

for block in list(model.children())[1:-3]: # Ignores first and last conv and bn
    block.layer[0].conv2.register_forward_hook(hook_fn)

print('Standard: ', compute_energy(model((torch.unsqueeze(x, 0)))).item())

print('Inner inputs ith block first stage: ', [torch.mean(outputs[t][0]).item() for t in outputs.keys()])
