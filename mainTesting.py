import torch
import os
import torchvision.transforms as transforms
import numpy as np
import torchattacks
import matplotlib.pyplot as plt
import sys
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from robustbench.utils import load_model

from energyUtils import compute_energy, compute_energyxy, test_pgd_impact
from models.model import NormalizedConfigurableModel
import models.model_cifar
from models.model_cifar import NormalizedWideResNet

def test_loop(model, dataloader, attack, energies, energies_xy, adv_energies, adv_energiesxy, adv, n_batches):
	'''
	This should be tweaked before being used again	
	'''
	# TODO with torch.no_grad():
	for batch, (X, y) in enumerate(dataloader):
		if batch <=  n_batches:
			pred = model(X)
			energies.extend(compute_energy(pred).detach().numpy())
			energies_xy.extend(compute_energyxy(pred, y).detach().numpy())
			adv = True
			if adv: 
				perturbed = attack(X, y)
				adv_logits = model(perturbed)
				adv_energies.extend(compute_energy(adv_logits).detach().numpy())
				adv_energiesxy.extend(compute_energyxy(adv_logits, y).detach().numpy())
		else:
			break


def main():

  # Change these if not in google collab
  sys.path.append('/content/drive/MyDrive/energyTesting/models')
  os.chdir('/content/drive/MyDrive/energyTesting')

  preprocess = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  device = ('cuda' if torch.cuda.is_available() else 'cpu')

  steps = [50]
  batch_size = 1024
  dataset = CIFAR10('./data', train=False, download=True, transform=preprocess)
  # when testing pgd impact on custom it's better to not shuffle to allow more
  # coherent saves
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(True if dataset == 'cuda' else False))
  
  # WRN 28-10
  #models = {'Standard' : '0.0%', 'Sehwag2020Hydra' : '57.14%', 'Wang2020Improving' : '56.29%', 'Zhang2020Geometry' : '59.64%',
  #              'Gowal2020Uncovering_28_10_extra' : '62.76%', 'Wang2023Better_WRN-28-10' : '67.31%'}
  
  # WRN 34-10
  models = {'Zhang2020Attacks' : '53.51%', 'Zhang2019You' : '44.83%', 'Wu2020Adversarial' : '56.17%',
            'Chen2024Data_WRN_34_10' : '57.30%', 'Addepalli2021Towards_WRN34' : '58.04%', 'Sehwag2021Proxy' : '60.27%',
            'Rade2021Helper_extra' : '62.83%'}

  print("\n" + "-*-" * 100 + "\n")
  for model_name in models:
    model_rob = models[model_name]
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    model.eval()
    # TODO: create this inside the function
    print(f"\nLoaded {model_name} successfully!")
    model = model.to(device)
    test_pgd_impact(steps, model, model_name, model_rob, dataloader, device)
    del model

  print("\n-*- Completed test suite!-*-" + '-*-' * 50)
  return


if __name__ == "__main__":
	main()
