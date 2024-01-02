import numpy as np
from matplotlib import pyplot as plt
import torch
import os

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
    ax.set_ylabel('Position in Bogenminuten [arcmin]')
    ax.set_xlabel('Zeit in Sekunden [s]')
    ax.set_title('Generierte Trajektorie - rescaled')
    ax.legend()

    fig.tight_layout()

    # Speichere die Figur
    fig.savefig(fr'{savepath}{i}.png')
    plt.close()

def plot_losses_and_mmd2(epoch,d_loss_log, g_loss_log,mmd2, savepath, filename):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Subplot für Discriminator und Generator Loss
    ax1.plot(np.linspace(0, epoch, len(d_loss_log)), d_loss_log, label='Diskriminatorverlust', color='blue')
    ax2 = ax1.twinx()  # Zweite Y-Achse für den rechten Plot
    ax2.plot(np.linspace(0, epoch, len(g_loss_log)), g_loss_log, label='Generatorverlust', color='red')

    ax1.set_ylabel('Diskriminatorverlust', color='blue', fontsize=16)
    ax2.set_ylabel('Generatorverlust', color='red', fontsize=16)

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
    ax3.set_xlabel('Epoche', fontsize=16)
    ax3.set_ylabel('MMD2', color='green', fontsize=16)
    ax3.tick_params('y', colors='green', labelsize=13)
    ax3.grid(True)

    # Layout verbessern
    plt.suptitle('Diskriminator- und Generatorverlust, MMD^2', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
    plt.savefig(fr'{savepath}', dpi=600)
    plt.close()

checkpoint_file = r"C:\Users\uvuik\Desktop\Torch\Roorda_scale=0.3,f=100,len=5s\RCGAN_Params_lr_0.0004_bs_48_hs_250_resample=100\Evaluation_2000.pth"
data = torch.load(checkpoint_file)
d_loss_log = data['d_loss_log']
g_loss_log= data['g_loss_log']
epoch = data['epoch']
mmd2 = data['mmd2']
print(mmd2[-1])
fft=data['fft_similarity']

savepath = r"C:\Users\uvuik\bwSyncShare\Bilder\GAN\Evaluation LR=0.0004, hs=250, resamp=100,5s.jpg"
filename = "BadTraining"
plot_losses_and_mmd2(epoch,d_loss_log,g_loss_log,mmd2,savepath, filename)

#filepath = r"\\os.lsdf.kit.edu\iai-projects\iai-aida\Daten_Anzlinger\AlleDaten_HAICORE\po1506-MA_Anzlinger\GazeBase0.3_f=100\RCGAN_Params_lr_0.0003_bs_128_hs_250_resample=100,len=5s\Evaluation_240.pth"
folderpath = r"\\os.lsdf.kit.edu\iai-projects\iai-aida\Daten_Anzlinger\AlleDaten_HAICORE\po1506-MA_Anzlinger\GazeBase0.3_f=100\RCGAN_Params_lr_0.0007_bs_128_hs_250_resample=100,len=5s"
filepaths = os.listdir(folderpath)
pth_files = [file for file in filepaths if file.endswith('.pth')]
for file in pth_files:
    filepath = os.path.join(folderpath, file)
    a = GenerateDataset(filepath)
    fsamp=100
    duration=5
    fsamp_out=100
    b = a.generate_data(a.model, 1, duration, fsamp, fsamp_out, labels=r"C:\Users\uvuik\Documents\Code\MasterarbeitIAI\GeneratingTraces_RGANtorch\FEM\GazeBaseLabels.csv", scalerfile=r'C:\\Users\\uvuik\\Documents\\Code\\MasterarbeitIAI\\GeneratingTraces_RGANtorch\\FEM\\ScalerRoorda5sf=100.save')
    savepath = filepath[:-4]
    plot_imgs(b[0], fsamp_out, "Trace")


#a = GenerateDataset(filepath)
#fsamp = 100
#duration = 5
#fsamp_out = 100
#b = a.generate_data(a.model, 1, duration, fsamp, fsamp_out, labels=r"C:\Users\uvuik\Documents\Code\MasterarbeitIAI\GeneratingTraces_RGANtorch\FEM\GazeBaseLabels.csv", scalerfile=r'C:\\Users\\uvuik\\Documents\\Code\\MasterarbeitIAI\\GeneratingTraces_RGANtorch\\FEM\\ScalerRoorda5sf=100.save')
#for i in range(0,4):
#    plot_imgs(b[i], fsamp_out, i)