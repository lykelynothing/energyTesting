import matplotlib.pyplot as plt
import os
import numpy as np
from energyUtils import parseTxt

''' Resnets No weights
means_path = './energyTesting/means/ResnetsRandom'
dict = parseTxt(means_path)
dict2 = parseTxt('./energyTesting/means/ResnetsRandom2')
dict3 = parseTxt('./energyTesting/means/ResnetsRandom3')
dict4 = parseTxt('./energyTesting/means/ResnetsRandom4')
dict5 = parseTxt('./energyTesting/means/ResnetsRandom5')
dict6 = parseTxt('./energyTesting/means/ResnetsRandom6')

steps = [1, 2, 5, 10, 20, 30, 50]

'''
dict = parseTxt('./means/WRN28-10RandMeaningful2', 0, '.txt')
steps = [1, 2, 4, 5, 10, 20]

plt.figure(figsize=(12, 8))

colors=['blue', 'orange', 'red', 'green']

for i, model in enumerate(dict):
    print(model)
    print(dict[model][2])
    plt.plot(steps, dict[model][2], label=model, marker='^')
    #for i, (x, y) in enumerate(zip(dict[model][2], dict[model][1])):
    #    plt.text(x, y, f'{steps[i]}', fontsize=6)

#plt.gca().add_patch(plt.Circle((-4, 40), 3, color='green', fill=False))

#plt.ylim(0, 10)
#plt.xlim(-1, 1)
#plt.xscale('symlog')
#plt.yscale('symlog')
plt.ylabel('Delta E(x)')
plt.xlabel('Steps')
plt.legend(loc='best')
plt.title('Pgd steps vs Delta E(x) WRN28-10 Random')
plt.grid(linewidth=0.3)
os.makedirs('./energyTesting/plots', exist_ok=True)
plt.savefig('./energyTesting/plots/final.pdf')
plt.show()