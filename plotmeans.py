import matplotlib.pyplot as plt
import os
import numpy as np
from energyUtils import parseAll


tup = parseAll('./means/ResNets/StdResnets')
dict = tup[3]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
steps = [1, 2, 5, 10, 20, 50]

for i, model in enumerate(dict):
    ax.plot(steps, dict[model], label=model, marker='^')

ax.set_ylabel('Delta E(x)')
ax.set_xlabel('PGD Steps')
ax.legend(loc='best')
ax.set_title('Delta E(x) over PGD steps')
ax.grid(linewidth=0.3)

os.makedirs('./plots/ResNets/StdResNets', exist_ok=True)
plt.savefig('./plots/ResNets/StdResNets/DeltaE(x)vsPGD.pdf', bbox_inches='tight')
plt.show()