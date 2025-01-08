import torch
import os
import time
import numpy as np
import math
from torch.utils.data import dataloader, Dataset
from typing import List
from torchattacks import PGD
from PIL import Image

class ImageNetSingleImage(Dataset):
		def __init__(self, dir, transform=None):
			self.dir = dir
			self.image_files = sorted([f for f in os.listdir(dir) if os.path.basename(f) != '.DS_Store'])
			self.transform = transform

		def __len__(self):
			return len(self.image_files)

		def __getitem__(self, index):
			img_path = os.path.join(self.dir, self.image_files[index])
			image = Image.open(img_path).convert("RGB")
			label = index

			if self.transform:
				image = self.transform(image)

			return image, label 



def test_pgd_impact(steps : List[int],
                model : torch.nn.Module, model_name : str, 
                dataloader : torch.utils.data.DataLoader, dir : str, model_rob: str = '0.0', 
                device : str = 'cpu') -> None :
    '''
    Takes a list of different steps of PGD to try, a model, a dataloader and
    a list where the mean energies for each kind of PGD step will be stored.
    Now also stores difference between mean normal energy and mean adversarial energy.
    '''
    
    for i in range(len(steps)):

      start_time = time.time()
      time_curr = 0.0
      time_prev = 0.0
      time_est = 0.0
      time_est_prev = 0.0

      acc_adv = 0
      mean_en = 0
      delta = 0

      attack = PGD(model, steps=steps[i])
      attack.set_normalization_used((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

      for batch, (X, y) in enumerate(dataloader):
        
        time_prev = time.time()

        X = X.to(device)
        y = y.to(device)
        adv = attack(X, y)
        
        with torch.no_grad():
            pred = model(X)
            pred_adv = model(adv)
        
        # compute mean energy of normal and adversarial logits and take  
        # their difference, accumulate and divide later
        delta += sum(compute_energy(pred).detach().cpu().numpy() - compute_energy(pred_adv).detach().cpu().numpy())
        
        # Accumulate all energy values and divide at the end
        mean_en += (sum(compute_energy(pred_adv).detach().cpu().numpy()))
        
        predicted_labels = torch.argmax(pred_adv, dim=1)
        matches = (y == predicted_labels)
        acc_adv += matches.sum().item()

        time_curr = time.time() - time_prev
        time_est = 0.9 * time_curr + 0.1 * time_est_prev
        time_est_prev = time_est

        bar = '#' * int(((batch + 1) / (len(dataloader))) * 20)
        print(f"\r| {model_name} | Current steps: {steps[i]} [{bar}] {( (batch + 1) / len(dataloader) ) * 100:.1f}% | Est. Time{(time_est * len(dataloader))/ 60: .2f} min | Elapsed {(time_prev - start_time) / 60 : .2f} min | Adv Acc: {(acc_adv / ((batch+1) * dataloader.batch_size)) * 100:.2f}% | Correct : {acc_adv} | Delta : {delta / ( (batch + 1) * dataloader.batch_size) :.2f}"
        , flush=True, end='')

      # change filepath accordingly
      os.makedirs(f"./means/{dir}", exist_ok=True)
      path = os.path.join(os.getcwd(), f"means/{dir}")
      with open(f"{path}/{model_name}_{model_rob}.txt", "a") as file:
        file.write(f"\nSteps {steps[i]} mean_en : ")
        file.write(str(mean_en / (len(dataloader) * dataloader.batch_size)) + '\n')
        file.write(f"Adversarial accuracy: {100 * acc_adv / (len(dataloader) * dataloader.batch_size) :.2f}%\n")
        file.write(f"Mean delta: {delta / (len(dataloader) * dataloader.batch_size)}\n")
        print('\n| * Saved values locally to ',  f"{path}/{model_name}_{model_rob}.txt")

      del attack

    return

def rand_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only initialize parameters that require gradients
            if 'conv' in name.lower():  # Convolutional layers
                torch.nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))  # Kaiming initialization for conv layers
            elif 'bn' in name.lower():  # BatchNorm layers
                if 'weight' in name:  # Gamma
                    torch.nn.init.ones_(param.data)
                elif 'bias' in name:  # Beta
                    torch.nn.init.zeros_(param.data)
            elif 'fc' in name.lower():  # Fully connected (Linear) layers
                if param.dim() == 2:  # Weights of the linear layer
                    torch.nn.init.xavier_uniform_(param.data)  # Xavier initialization
                elif param.dim() == 1:  # Bias of the linear layer
                    torch.nn.init.zeros_(param.data)  # Initialize bias to 0
            elif 'bias' in name:  # General bias terms
                torch.nn.init.constant_(param.data, 0.0)

def parseTxt(means_path, lines, end):
    '''
    Parses all txt files contained in means_path, creates a model where each
    key is the model of a .txt file and its value is a list of lists where
    dict[model] = [[mean_en], [adv_acc], [mean_delta]]
    
    Lines: controls how many initial lines of the .txt to skip when reading

    End: shared end of file to remove when extracting names of models
    '''
    dict = {}
    for filename in os.listdir(means_path):
        if filename.endswith('.txt'):
            name = filename.replace(end, '')
            #name = name[:-7] + ' ' + name[-6:]
            print('Found ', name)
            means = []
            acc = []
            mean_delta = []
            filepath = os.path.join(means_path, filename)
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                # currently skipping 10 lines because of old results
                    if i > lines:
                        if 'mean_en' in line:
                            mean_en = float(line.split(':')[1].strip())
                            means.append(mean_en)
                        elif 'Adversarial accuracy' in line:
                            adv_acc = float(line.split(':')[1].strip().replace('%', ''))
                            acc.append(adv_acc)
                        elif 'Mean delta' in line:
                            delta = float(line.split(':')[1].strip())
                            mean_delta.append(delta)
            dict[name] = [means, acc, mean_delta]
    return dict

# standard energy
def compute_energy(logits):
	energy = -torch.logsumexp(logits, dim=1)
	return energy

# returns joint energy of predicted label
def compute_energyxy(logits, labels):
	correct_logits = logits[torch.arange(logits.size(0)), labels]
	energy = -correct_logits
	return energy

