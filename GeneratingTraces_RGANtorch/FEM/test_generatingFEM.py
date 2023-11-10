import numpy as np
from matplotlib import pyplot as plt
import torch

from GeneratingTraces_RGANtorch.FEM.generate_dataset import GenerateDataset

savepath = r"C:\Users\uvuik\Desktop\Torch\Bild"
def plot_imgs(data, f, i):
    frequency=f
    x1 = data['x']
    y1 = data['y']

    # Erstelle den Plot
    fig, ax = plt.subplots()

    # Plotte beide Koordinaten in einem einzigen Plot
    time_axis = np.arange(0, len(x1) / frequency, 1 / frequency)
    ax.plot(time_axis, x1, label='x-Koordinate')
    ax.plot(time_axis, y1, label='y-Koordinate')

    # Setze Achsenbeschriftungen und Legend
    ax.set_ylabel('Koordinaten')
    ax.set_xlabel('Zeit in Sekunden [s]')
    ax.legend()

    fig.tight_layout()

    # Speichere die Figur
    fig.savefig(fr'{savepath}{i}.png')
    plt.close()

    return True
checkpoint_file = r"C:\Users\uvuik\Desktop\Torch\Roorda\RCGAN_Params_lr_0.005_bs_32_hs_150\Evaluation_1490.pth"
best_epoch = torch.load(checkpoint_file)['best_epoch']
filepath = r"C:\Users\uvuik\Desktop\Torch\RCGAN_RandomLabels_DlossDouble\Evaluation_1680.pth"
a = GenerateDataset(filepath)
fsamp = 250
duration = 5
b = a.generate_data(a.model, 10, duration, fsamp, 250, labels=r"C:\Users\uvuik\Documents\Code\MasterarbeitIAI\GeneratingTraces_RGANtorch\FEM\RoordaLabels.csv")
for i in range(0,4):
    plot_imgs(b[i], fsamp, i)