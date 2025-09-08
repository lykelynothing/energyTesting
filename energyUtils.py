import torch
import datetime
import csv
import subprocess
import os
import time
import numpy as np
import math
import torch.linalg as la
from torch.nn.utils import prune
from torch.utils.data import Dataset, DataLoader
from typing import List
from torchattacks import PGD
from PIL import Image
from multiprocessing import Pool
from functools import partial
from scipy.optimize import fmin
from scipy.stats import kstest, weibull_min

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


def test_pgd_impact(
    steps: List[int],
    model: torch.nn.Module,
    model_name: str,
    dataloader: torch.utils.data.DataLoader,
    model_rob: str = '0.0',
    device: str = 'cpu',
    alpha: float = 2/255,
    eps: float = 8/255,
    n_it=10
) -> None:
    '''
    Takes a list of different steps of PGD to try, a model, a dataloader and
    a list where the mean energies for each kind of PGD step will be stored.
    Now also stores difference between mean normal energy and mean adversarial energy.
    n_it is the fraction of the dataset that will actually be tested.
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
        delta_xy = 0
        mean_xy = 0

        attack = PGD(model, steps=steps[i])
        attack.set_normalization_used(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )

        for batch, (X, y) in enumerate(dataloader):
            if batch == len(dataloader) // n_it:
                break
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
            en_xy = sum(compute_energyxy(pred, y).detach().cpu().numpy())
            adv_en_xy = sum(compute_energyxy(pred_adv, y).detach().cpu().numpy())

            mean_xy += adv_en_xy
            delta_xy += en_xy - adv_en_xy

            predicted_labels = torch.argmax(pred_adv, dim=1)
            matches = (y == predicted_labels)
            acc_adv += matches.sum().item()

            time_curr = time.time() - time_prev
            time_est = 0.9 * time_curr + 0.1 * time_est_prev
            time_est_prev = time_est

            tot_samples = len(dataloader) * dataloader.batch_size / n_it
            progress = (batch + 1) / (len(dataloader) / n_it)

            bar = '#' * int(progress * 20)
            print(
                f"\r| {model_name} | Current steps: {steps[i]} [{bar}] {progress * 100:.1f}% "
                f"| Est. Time {(time_est * len(dataloader)/n_it) / 60:.2f} min "
                f"| Elapsed {(time_prev - start_time) / 60:.2f} min "
                f"| Adv Acc: {(acc_adv / ((batch+1) * dataloader.batch_size)) * 100:.2f}% "
                f"| Correct : {acc_adv} "
                f"| Delta : {delta / ((batch + 1) * dataloader.batch_size):.4f} "
                f"| Mean : {mean_en / ((batch + 1) * dataloader.batch_size):.3f}\n"
                f"Mean Normal : {en_x / ((batch + 1) * dataloader.batch_size):.3f} "
                f"| Normal xy : {en_xy / ((batch + 1) * dataloader.batch_size):.3f}",
                flush=True,
                end=''
            )

        acc_adv = (100 * acc_adv * n_it) / (len(dataloader) * dataloader.batch_size)
        mean_en = mean_en / tot_samples
        delta = (delta * n_it) / (len(dataloader) * dataloader.batch_size)
        mean_xy = (mean_xy * n_it) / (len(dataloader) * dataloader.batch_size)
        en_x = (en_x * n_it) / (len(dataloader) * dataloader.batch_size)
        en_xy = (en_xy * n_it) / (len(dataloader) * dataloader.batch_size)
        delta_xy = (delta_xy * n_it) / (len(dataloader) * dataloader.batch_size)

        data = [
            steps[i], acc_adv, delta,
            mean_xy, mean_en, en_x, en_xy, delta_xy
        ]

        save_pgt_csv(data, model_name)

        del attack

    return


def save_pgt_csv(data, model_name):
    '''
    Saves results of test_pgd_impact on a csv file
    Add Model name in fields!!
    Also specify how folders will be named and organized
    '''
    fieldnames = ['Steps', 'Adversarial accuracy', 'Mean delta', 'Mean xy', 'Mean normal energy'
                  'Mean normal xy', 'Delta xy']
    os.makedirs(f"./pgd_impact/{str(datetime.date.today())}", exist_ok=True)
    path = os.path.join(os.getcwd(), f"./pgd_impact/{str(datetime.date.today())}/{model_name}.csv")
    with open(path, mode='a') as f:
        writer = csv.writer(f, delimiter=' , ')
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            writer.writerow(fieldnames)
        writer.writerow(data)
    
    print('\n| * Saved values locally to ',  f"./pgd_impact/{str(datetime.date.today())}/{model_name}.csv")
    return


def extract_kernels(model):
    kernels = []
    for n, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if 'conv' in n.lower() or 'shortcut' in n.lower() and 'weight' in name:
                kernels.append(param.data)
    return kernels

def rand_weights(model, inplace : bool = True, track : bool = True):
    for n, module in model.named_modules(): 
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                if 'conv' in n.lower() or 'shortcut' in n.lower() and 'weight' in name:  # Check if conv
                    #torch.nn.init.normal_(param.data)
                    torch.nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='relu')  
                elif 'bias' in name:  # For bias parameters
                    torch.nn.init.constant_(param.data, 0.0)
                elif 'bn' in n.lower() or 'batchnorm' in n.lower():  # BatchNorm layer
                    if 'weight' in name:  # Gamma
                        torch.nn.init.ones_(param.data)
                    elif 'bias' in name:  # Beta
                        torch.nn.init.zeros_(param.data)
                    if not track: # Set flag of bn to not use stored stats
                        module.track_running_stats = False
                elif 'downsample' in n.lower(): # some custom wrns have this
                    if '1' in n.lower(): # if 1 is in name it's a BatchNorm
                        if 'bias' in name.lower():
                            torch.nn.init.zeros_(param.data)
                        else:
                            torch.nn.init.ones_(param.data)
                    elif '0' in n.lower():
                        torch.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                    else:
                        print("No param condition: ", name, n) # To check if some parameters weren't caught
                elif 'weight' in name.lower() and 'fc' in n or 'linear' in n or 'logits' in n:
                    #torch.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.001)
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
    dict[model] = [means, acc, mean_delta, mean_xy, mean_delta_xy]
    
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
            mean_delta_xy = []
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
                        elif 'Delta xy' in line:
                            delta_xy = float(line.split(':')[1].strip())
                            mean_delta_xy.append(delta_xy)
            dict[name] = [means, acc, mean_delta, mean_xy, mean_delta_xy]
    return dict

# like parseTxt but only reads mean delta E(x, y)
def parseTxtXY(path):
    means = {}
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            mean_delta_xy = []
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    if 'Delta xy' in line:
                        delta_xy = float(line.split(':')[1].strip())
                        mean_delta_xy.append(delta_xy)
            if '_0.0.txt' in filename:
                filename = filename.replace('_0.0.txt', '')
            means[filename] = mean_delta_xy
    return means

'''
 Takes a folder containing the results of multiple tests
 accumulates values of mean delta E(x, y), delta E(x), mean E(x) for each 
 corresponding pgd step.
 also computes variance (using n-1 degrees of freedom) of delta E(x, y)
'''
def parseAll(path : str) -> tuple:
    '''
    :return:
        - tot_means 
        - tot_vars 
        - tot_deltas_xy
        - tot_deltas_x
        - tot_means_xy
    '''
    tot_means = {}
    tot_deltas_x = {}
    tot_deltas_xy = {}
    tot_vars = {}
    tot_means_xy = {}
    trials = 0 # to count number of trials
    i = 6 # number of pgd steps
    # initializes dict with correct names as keys
    for name in os.listdir(os.path.join(path, os.listdir(path)[0])):
        name = name.replace('_0.0.txt', '') # will surely create problems
        tot_deltas_xy[name] = np.array([0.0] * i) # update this so that it's long as many pgd steps as tested
        tot_vars[name] = np.array([0.0] * i)
        tot_means[name] = np.array([0.0] * i)
        tot_deltas_x[name] = np.array([0.0] * i)
        tot_means_xy[name] = np.array([0.0] * i)
    for dirname in os.listdir(path):
        print(f"Checking {dirname}")
        # Visits each trial subfolder, accumulates values from it in tot_mean
        if os.path.isdir(os.path.join(path, dirname)):
            trials += 1
            dir = os.path.join(path, dirname)
            res = parseTxt(dir, 0, '_0.0.txt')
            # res contains dict with key = model and value = list of values for each pgd step
            for model in res:
                # for each model iterate over its lists of lists of values
                tot_means[model] += np.array(res[model][0])
                tot_deltas_xy[model] += np.array(res[model][4])
                tot_deltas_x[model] += np.array(res[model][2])
                tot_means_xy[model] += np.array(res[model][3])
    # Then divides by number of trials
    for model in tot_deltas_xy:
        tot_deltas_xy[model]  = tot_deltas_xy[model] / trials # can be done because it's an np array
        tot_deltas_x[model] = tot_deltas_x[model] / trials
        tot_means[model] = tot_means[model] / trials
        tot_means_xy[model] = tot_means_xy[model] / trials
    # Vars must be computed afterwards since we need exact means first
    tot_vars = compute_Var(tot_vars, tot_deltas_xy, path)

    # Compute std from vars
    for model in tot_vars:
        for i in range(len(tot_vars[model])):
            tot_vars[model][i] = math.sqrt(tot_vars[model][i])

    return tot_means, tot_vars, tot_deltas_xy, tot_deltas_x, tot_means_xy

# Accumulates variance values
def compute_Var(tot_vars : dict, tot_means : dict, path : str):
    for dirname in os.listdir(path):
        dir = os.path.join(path, dirname)
        if os.path.isdir(dir):
            res = parseTxtXY(dir)
            for model in res:
                for i in range(len(res[model])):
                    tot_vars[model][i] += math.pow((res[model][i] - tot_means[model][i]), 2)
    return tot_vars

# standard energy
def compute_energy(logits):
	energy = -torch.logsumexp(logits, dim=1)
	return energy

# returns joint energy of ground truth label
def compute_energyxy(logits, labels):
	correct_logits = logits[torch.arange(logits.size(0)), labels]
	energy = -correct_logits
	return energy

def cross_lip(input, logits : torch.Tensor, model : torch.nn.Module, n_samples, epsilon, j, c):
    '''
    Following Evaluating the robustness.., computes differences between max logit
    and all other logit values. Then estimates lipschitz constant of difference of
    pairs of logits, and from there estimates its maximum value with extreme value theory.
    Higher cross lip value ~ higher robustess.
    We generate in batches multiple points around input, and then consider maximum
    gradient norm of each batch. Then we do MLE.
    '''
    model.eval()
    # TODO: compute norm of gradient also for unperturbed input

    # Generate for each delta samples around p-ball to estimate max grad
    for n in range(n_samples):
        pert = torch.empty_like(input).uniform(-epsilon, epsilon)
        sample = input + pert
        new_logits = model(sample)
        new_pred = torch.max(new_logits, dim=1)
        # then for each logit pair compute maximum gradient norm

def cross_lip_t(input : torch.Tensor, model : torch.nn.Module, c : int, j: int, epsilon : float,  n_samples : int = 1024, batches : int = 500):
    '''
    Same as cross_lip but only considers one pair of logits.
    '''
    # Clone to be sure all gradients are tracked
    x = input.clone().detach()
    grads = []
    for b in range(batches):
        start = time.time()
        batch_norms = []

        perts = torch.empty((n_samples, *x.shape[1:]), device=x.device).uniform_(-epsilon, epsilon)
        samples = x + perts
        samples.requires_grad_(True)
        print("Samples shape: ", samples.shape)
        logits = model(samples)
        differences = logits[:, c] - logits[:, j]
        # The following is MAJOR bullshit 
        for i in range(n_samples):
            grad_diffs_all = torch.autograd.grad(differences[i], samples, retain_graph=True)[0]
            batch_norms.append(grad_diffs_all[i].norm(p=float('inf')))
        print("Norms shape: ", batch_norms.shape)

        for n in range(n_samples):
            # This takes approximately a minute
            # Generate sample
            # TODO: check for better sampling method
            pert = torch.empty_like(input).uniform_(-epsilon, epsilon)
            sample = x + pert
            logits = model(sample)
            difference = logits[0, c] - logits[0, j]
            grad_diff = torch.autograd.grad(difference, x)[0]
            norm = grad_diff.norm(p=float('inf'))
            batch_grads.append(norm)
        grads.append(max(batch_grads))
        elapsed = time.time() - start
        print(f"Batch {b+1}/{batches} done in {elapsed:.2f} sec. Current max grad: {max(grads):.4f}, estimated time left: {(batches - b - 1) * elapsed / 60:.2f} min")
    # MLE of grads
    _, c, loc, sacle, ks, pVal = weibull_mle(torch.tensor(grads, device='cpu').numpy())
    return -loc # -loc is lip estimate


def weibull_mle(sample):
    # I don't really understand this code
    c_init = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    pool = Pool(processes= len(c_init) + 1)
    # dict to save fitted params
    fitted_paras = {"c" : [], "loc" : [], "scale" : [], "ks" : [], "pVal" : []}
    loc_shift = np.max(sample)
    dist_range = loc_shift - np.min(sample)
    shape_rescale = dist_range
    rescaled_sample = np.copy(sample)
    rescaled_sample -= loc_shift
    rescaled_sample /= shape_rescale

    results = pool.map(partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale, fmin), c_init)

    for res, c_i in zip(results, c_init):
        c = res[0]
        loc = res[1]
        scale = res[2]
        ks = res[3]
        pVal = res[4]
        print("[DEBUG][L2] c_init = {:5.5g}, fitted c = {:6.2f}, loc = {:7.2f}, scale = {:7.2f}, ks = {:4.2f}, pVal = {:4.2f}, max = {:7.2f}".format(c_i,c,loc,scale,ks,pVal,loc_shift))
        
        ## plot every fitted result
        #plot_weibull(sample,c,loc,scale,ks,pVal,p)
        
        fitted_paras['c'].append(c)
        fitted_paras['loc'].append(loc)
        fitted_paras['scale'].append(scale)
        fitted_paras['ks'].append(ks)
        fitted_paras['pVal'].append(pVal)
    
    
    # get the paras of best pVal among c_init
    max_pVal = np.nanmax(fitted_paras['pVal'])
    if np.isnan(max_pVal) or max_pVal < 0.001:
        print("ill-conditioned samples. Using maximum sample value.")
        # handle the ill conditioned case
        return -1, -1, -max(sample), -1, -1, -1

    max_pVal_idx = fitted_paras['pVal'].index(max_pVal)
    
    c_init_best = c_init[max_pVal_idx]
    c_best = fitted_paras['c'][max_pVal_idx]
    loc_best = fitted_paras['loc'][max_pVal_idx]
    scale_best = fitted_paras['scale'][max_pVal_idx]
    ks_best = fitted_paras['ks'][max_pVal_idx]
    pVal_best = fitted_paras['pVal'][max_pVal_idx]
    
    return c_init_best, c_best, loc_best, scale_best, ks_best, pVal_best


def fit_and_test(rescaled_sample, sample, loc_shift, shape_rescale, optimizer, c_i):
    [c, loc, scale] = weibull_min.fit(-rescaled_sample, c_i, optimizer=optimizer)
    loc = -loc_shift + loc * shape_rescale
    scale *= shape_rescale
    ks, pval = kstest(-sample, 'weibull_min', args=(c, loc, scale))
    return c, loc, scale, ks, pval


def git_update():
    try:
        subprocess.run(['git', 'status'], check=True, capture_output=True)

        git_status = subprocess.run(
            ["git", "status", "--porcelain", './means'],
            check=True,
            stdout=subprocess.PIPE
        )
        changes = git_status.stdout.decode("utf-8").strip()

        if not changes:
            return f"No changes detected in folder ./means"

        subprocess.run(['git', 'add', './means'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Updated means'], check=True, capture_output=True)
        res = subprocess.run(['git', 'push'], check=True, capture_output=True)
        return res.stdout.decode('utf-8')

    except subprocess.CalledProcessError as e:
        return f"Git error: {e.stderr.decode('utf-8')}" if e.stderr else str(e) 
    except Exception as e:
         return f"Error: {str(e)}"
    
def fit_image(model, image, y, n=500):
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    running_loss = 0
    for i in range(n):
        opt.zero_grad()
        out = model(image)
        loss = loss_fn(out, y)
        running_loss += loss.item()
        loss.backward()
        opt.step()
        if (i+1) % 50 == 0:
            print(f"Current loss : {running_loss / i : .3f}")

def collect_norms(model : torch.nn.Module, norm : str = 'infty') -> list:
    """
    Computes specified norm of weight matrices for all layers of model
    and collects them in a list.

    Args:
        model (torch.nn.Module): pytorch model

        norm (str) : norm to use
    
    Returns:
        norms (list[float]) : list of norms for each weight matrix

    """
    norms = []



    return norms

def collect_z_norms(model : torch.nn.Module, dataloader : DataLoader, norm = float('inf'), n_it : int = 10, 
                    device : str = 'cpu') -> float :
    """
    Computes average specified norm of logits of model over
    a dataset contained in dataloader

    Args:
        model (torch.nn.Module) : pytorch model

        dataloader (DataLoader) : torch data loader

        norm : norm to use on logits

        n_it (int) : number of batches to process
    Returns:
        mean_norm (float) : mean specified norm over whole dataset

    """

    model.eval()
    
    current_norm = 0.0

    with torch.no_grad():
        for batch, (X, _) in enumerate(dataloader):
            if batch == n_it:
                break
            
            X = X.to(device)
            pred = model(X)
            # Compute mean per batch
            batch_norm = torch.mean(la.vector_norm(pred, ord=norm, dim=(-1))).detach()
            current_norm += batch_norm
            
            if batch % (n_it // 4) == 0:
                print(f'Currently at batch {batch}')

    mean_norm = (current_norm) / (n_it)

    return mean_norm

def save_z_norms(path, data):

    fieldnames = ['model', 'logit_norm'] 
    os.makedirs(f"./logit_norms/{path}", exist_ok=True)
    path = os.path.join(os.getcwd(), f"./logit_norms/{path}/logits.csv")
    with open(path, mode='a') as file: # add exact path
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(data) 
    return