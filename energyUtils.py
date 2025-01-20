import torch
import os
import time
import numpy as np
import math
from torch.nn.utils import prune
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

def check_pruning(model):
    for name, module in model.named_modules():
        if prune.is_pruned(module):
            print(f"Layer '{name}' is pruned.")
        else:
            print(f"Layer '{name}' is not pruned.")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_pgd_impact(steps : List[int],
                model : torch.nn.Module, model_name : str, 
                dataloader : torch.utils.data.DataLoader, dir : str, model_rob: str = '0.0', 
                device : str = 'cpu', alpha : float = 2/255, eps : float = 8/255) -> None :
    '''
    Takes a list of different steps of PGD to try, a model, a dataloader and
    a list where the mean energies for each kind of PGD step will be stored.
    Now also stores difference between mean normal energy and mean adversarial energy.
    '''
    
    model.loss_fn = torch.nn.CrossEntropyLoss()


    for i in range(len(steps)):

      start_time = time.time()
      time_curr = 0.0
      time_prev = 0.0
      time_est = 0.0
      time_est_prev = 0.0

      acc_adv = 0
      mean_en = 0
      delta = 0
      mean_xy = 0    

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
        
        en_x = sum(compute_energy(pred).detach().cpu().numpy())
        adv_en_x = sum(compute_energy(pred_adv).detach().cpu().numpy())

        # compute mean energy of normal and adversarial logits and take  
        # their difference, accumulate and divide later
        delta += en_x - adv_en_x
        
        # Accumulate all energy values and divide at the end
        mean_en += adv_en_x

        # Compute mean xy energy
        en_xy = (sum(compute_energyxy(pred, y).detach().cpu().numpy()))
        adv_en_xy = (sum(compute_energyxy(pred_adv, y).detach().cpu().numpy()))
        mean_xy += adv_en_xy 

        predicted_labels = torch.argmax(pred_adv, dim=1)
        matches = (y == predicted_labels)
        acc_adv += matches.sum().item()

        time_curr = time.time() - time_prev
        time_est = 0.9 * time_curr + 0.1 * time_est_prev
        time_est_prev = time_est

        bar = '#' * int(((batch + 1) / (len(dataloader))) * 20)
        print(f"\r| {model_name} | Current steps: {steps[i]} [{bar}] {( (batch + 1) / len(dataloader) ) * 100:.1f}% | Est. Time{(time_est * len(dataloader))/ 60: .2f} min" 
              + f"| Elapsed {(time_prev - start_time) / 60 : .2f} min | Adv Acc: {(acc_adv / ((batch+1) * dataloader.batch_size)) * 100:.2f}%" 
              + f"| Correct : {acc_adv} | Delta : {delta / ( (batch + 1) * dataloader.batch_size) :.4f} | Mean : {mean_en / ( (batch + 1) * dataloader.batch_size)}"
              + f"\nMean Normal : {en_x/ ( (batch + 1) * dataloader.batch_size)} | Normal xy : {en_xy / ( (batch + 1) * dataloader.batch_size)}"
        , flush=True, end='')

      # change filepath accordingly
      os.makedirs(f"./means/{dir}", exist_ok=True)
      path = os.path.join(os.getcwd(), f"means/{dir}")
      with open(f"{path}/{model_name}_{model_rob}.txt", "a") as file:
        file.write(f"\nSteps {steps[i]} mean_en : ")
        file.write(str(mean_en / (len(dataloader) * dataloader.batch_size)) + '\n')

        file.write(f"Adversarial accuracy: {100 * acc_adv / (len(dataloader) * dataloader.batch_size) :.2f}%\n")
        file.write(f"Mean delta: {delta / (len(dataloader) * dataloader.batch_size)}\n")
        file.write(f"Mean xy: {mean_xy / (len(dataloader) * dataloader.batch_size)}\n")
        file.write(f"Mean Normal Energy : {en_x / (len(dataloader) * dataloader.batch_size)}\n")
        file.write(f"Mean Normal E xy : {en_xy / (len(dataloader) * dataloader.batch_size)}\n")
        print('\n| * Saved values locally to ',  f"{path}/{model_name}_{model_rob}.txt")

      del attack

    return

def rand_weights(model, inplace : bool = True, track : bool = True):
    for n, module in model.named_modules(): 
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                if 'conv' in n.lower() and 'weight' in name:  # Check if conv
                    #torch.nn.init.normal_(param.data)
                    torch.nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='relu')  
                elif 'bias' in name:  # For bias parameters
                    torch.nn.init.constant_(param.data, 0.0)
                elif 'bn' in n.lower():  # BatchNorm layer
                    if 'weight' in name:  # Gamma
                        torch.nn.init.ones_(param.data)
                    elif 'bias' in name:  # Beta
                        torch.nn.init.zeros_(param.data)
                    if not track: # index modules dict to find right module and change its flag
                        module.track_running_stats = False
                elif 'downsample' in name.lower(): # some custom wrns have this
                    if '1' in name.lower():
                        if 'bias' in name.lower():
                            torch.nn.init.zeros_(param.data)
                        else:
                            torch.nn.init.ones_(param.data)
                    else:
                        torch.nn.init.normal_(param.data)
                        #torch.nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='relu')
                elif 'weight' in name.lower() and ('fc' or 'linear') in n:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
                else:
                    print("No param condition: ", name, n) # To check if some parameters weren't caught
            else:
                 print('No grad: ', name, n)
        if 'relu' in n.lower(): # act_fns should go here, in general all parms with no req_grad
            module.inplace = inplace


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
            mean_xy = []
            filepath = os.path.join(means_path, filename)
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                # currently skipping 10 lines because of old results
                    if i >= lines:
                        if 'mean_en' in line:
                            mean_en = float(line.split(':')[1].strip())
                            means.append(mean_en)
                        elif 'Adversarial accuracy' in line:
                            adv_acc = float(line.split(':')[1].strip().replace('%', ''))
                            acc.append(adv_acc)
                        elif 'Mean delta' in line:
                            delta = float(line.split(':')[1].strip())
                            mean_delta.append(delta)
                        elif 'Mean xy' in line:
                            xy = float(line.split(':')[1].strip())
                            mean_xy.append(xy)
            dict[name] = [means, acc, mean_delta, mean_xy]
    return dict

# standard energy
def compute_energy(logits):
	energy = -torch.logsumexp(logits, dim=1)
	return energy

# returns joint energy of ground truth label
def compute_energyxy(logits, labels):
	correct_logits = logits[torch.arange(logits.size(0)), labels]
	energy = -correct_logits
	return energy

