import torch
import numpy as np
from lipschitz_bound.lipschitz_bound import LipschitzBound

kernel_numpy = np.random.randn(1, 3, 3, 3)
kernel_torch = torch.FloatTensor(kernel_numpy)
lb = LipschitzBound(kernel_numpy.shape, padding=1, sample=50, backend='torch', cuda=False)
sv_bound = lb.compute(kernel_torch)
print(f'LipBound : {sv_bound:.3f}')

