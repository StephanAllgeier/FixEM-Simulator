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

def plot_losses_and_mmd2(epoch,d_loss_log, g_loss_log,mmd2, savepath, filename):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Subplot für Discriminator und Generator Loss
    ax1.plot(np.linspace(0, epoch, len(d_loss_log)), d_loss_log, label='Discriminator Loss', color='blue')
    ax2 = ax1.twinx()  # Zweite Y-Achse für den rechten Plot
    ax2.plot(np.linspace(0, epoch, len(g_loss_log)), g_loss_log, label='Generator Loss', color='red')

    ax1.set_ylabel('Discriminator Loss', color='blue', fontsize=16)
    ax2.set_ylabel('Generator Loss', color='red', fontsize=16)

    ax1.tick_params('y', colors='blue', labelsize=13)
    ax2.tick_params('y', colors='red', labelsize=13)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.grid(True)
    ax2.grid(True)
    min_y = 0
    max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(min_y, max_y)
    ax2.set_ylim(min_y, max_y)


    # Subplot für MMD2 (zweite Zeile des Plots)
    ax3.plot(np.linspace(0, epoch, len(mmd2)), mmd2, label='MMD2', color='green')
    ax3.set_xlabel('Epoch', fontsize=16)
    ax3.set_ylabel('MMD2', color='green', fontsize=16)
    ax3.tick_params('y', colors='green', labelsize=13)
    ax3.grid(True)

    # Layout verbessern
    plt.suptitle('Discriminator and Generator Loss, MMD2 Over Time', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
    plt.savefig(fr'{savepath}\{filename}.jpg', dpi=600)
    plt.close()

checkpoint_file = r"C:\Users\uvuik\Desktop\Torch\Roorda_scale=0.1,f=100\RCGAN_Params_lr_0.0011_bs_128_hs_50_resample=250\Evaluation_2990.pth"
data = torch.load(checkpoint_file)
d_loss_log = data['d_loss_log']
g_loss_log= data['g_loss_log']
epoch = data['epoch']
mmd2 = data['mmd2']
savepath = r"C:\Users\uvuik\Desktop\Torch Ergebnisse\Bilder"
filename = "BadTraining"
plot_losses_and_mmd2(epoch,d_loss_log,g_loss_log,mmd2,savepath, filename)
filepath = r"C:\Users\uvuik\Desktop\Torch\RCGAN_RandomLabels_DlossDouble\Evaluation_1680.pth"
a = GenerateDataset(filepath)
fsamp = 250
duration = 5
b = a.generate_data(a.model, 10, duration, fsamp, 250, labels=r"C:\Users\uvuik\Documents\Code\MasterarbeitIAI\GeneratingTraces_RGANtorch\FEM\RoordaLabels.csv")
for i in range(0,4):
    plot_imgs(b[i], fsamp, i)