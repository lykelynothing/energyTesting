import matplotlib.pyplot as plt
import os

means_path = './energyTesting/means/WRN34-10'
dict = {}

for filename in os.listdir(means_path):
    if filename.endswith('.txt'):
        name = filename.replace('_WideResNet-34-10.txt', '%')
        name = name[:-7] + ' ' + name[-6:]
        means = []
        acc = []
        mean_delta = []
        filepath = os.path.join(means_path, filename)
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
            # currently skipping 10 lines because of old results
                if i > 10:
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

steps = [1, 2, 5, 10, 20, 50]

plt.figure(figsize=(10, 8))


for model in dict:
    plt.plot(dict[model][2], dict[model][1], marker='^', label=model)
    for i, (x, y) in enumerate(zip(dict[model][2], dict[model][1])):
        plt.text(x, y, f'{steps[i]}', fontsize=6)


#plt.gca().add_patch(plt.Circle((-4, 40), 3, color='green', fill=False))

#plt.ylim(0, 10)
#plt.xlim(-1, 1)
plt.xscale('symlog')
#plt.yscale('symlog')
plt.xlabel('Delta E(x)')
plt.ylabel('Robust accuracy %')
plt.legend()
plt.title('Comparing E(x) to robust accuracy WRN34-10')
plt.grid(linewidth=0.3)
os.makedirs('./plots', exist_ok=True)
plt.savefig('./plots/AccVsDelta34-10.pdf')
plt.show()