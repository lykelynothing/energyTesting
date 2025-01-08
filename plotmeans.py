import matplotlib.pyplot as plt
import os
from energyUtils import parseTxt

means_path = './means/WRN28-10Rand'

dict = parseTxt(means_path, 0, '.txt')

steps = [1, 2, 5]

plt.figure(figsize=(10, 8))

for i, model in enumerate(dict):
    plt.plot(steps, dict[model][2], marker='^', label=model)
#    for i, (x, y) in enumerate(zip(dict[model][2], dict[model][1])):
#        plt.text(x, y, f'{steps[i]}', fontsize=6)


#plt.gca().add_patch(plt.Circle((-4, 40), 3, color='green', fill=False))

#plt.ylim(0, 10)
#plt.xlim(-1, 1)
#plt.xscale('symlog')
plt.yscale('symlog')
plt.xlabel('PGD steps')
plt.ylabel('Delta E(x)')
plt.legend()
plt.title('Comparing pgd steps to delta E(x) Resnets Std')
plt.grid(linewidth=0.3)
os.makedirs('./plots', exist_ok=True)
plt.savefig('./plots/StepsVsDeltaE(x)WRNRandom.pdf')
plt.show()