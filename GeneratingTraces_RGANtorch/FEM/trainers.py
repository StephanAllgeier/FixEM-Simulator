"""
References:
    - https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
"""
import torch
import numpy as np
import torch.nn as nn
import torchgan
from matplotlib.ticker import StrMethodFormatter
from torch.nn import BCELoss, BCEWithLogitsLoss
from GeneratingTraces_RGANtorch.FEM import make_logger
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
import GeneratingTraces_RGANtorch.FEM.mmd as mmd
import matplotlib.pyplot as plt
logger = make_logger(__file__)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.fft import fft

class SequenceTrainer:
    def __init__(self, models, recon, ncritic, losses_list, epochs, retain_checkpoints, checkpoints, mlflow_interval, device, noise_size, vali_set, savepath, GANtype, resamp_frequency, eval_frequency=1, conv_window = 10, scale=None):
        self.models = models
        self.recon = recon
        self.ncritic = ncritic
        self.losses_list = losses_list
        self.epochs = epochs
        self.retain_checkpoints = retain_checkpoints
        self.checkpoints = checkpoints
        self.mlflow_interval = mlflow_interval
        self.device = device
        self.noise_size= noise_size
        self.g_loss_log_epoch = []
        self.d_loss_log_epoch = []
        self.g_loss_log = []
        self.d_loss_log = []
        self.mmd2 = []
        self.fft_similarity = []
        self.best_mmd2 = 1000
        self.vali_set = vali_set
        self.eval_frequency = eval_frequency
        self.best_epoch = 0
        self.savepath = savepath
        self.GANtype = GANtype
        self.conv_window = conv_window
        self.scale=scale
        self.frequency = resamp_frequency

        # Erstelle Generator und Diskriminator hier
        self.generator = self.models['generator']['name'](**self.models['generator']['args']).to(self.device)
        self.discriminator = self.models['discriminator']['name'](**self.models['discriminator']['args']).to(
            self.device)

        # Erstelle Optimizer für Generator und Diskriminator
        self.optimizer_g = self.models['generator']['optimizer']['name'](self.generator.parameters(),
                                                                         **self.models['generator']['optimizer'][
                                                                             'args'])
        self.optimizer_d = self.models['discriminator']['optimizer']['name'](self.discriminator.parameters(),
                                                            **self.models['discriminator']['optimizer']['args'])

    def on_epoch_end(self, epoch, fake_labels=None):
        #Gloss und d_loss berechnen und anschließend die liste zurücksetzen
        self.g_loss_log.append(sum(self.g_loss_log_epoch) / len(self.g_loss_log_epoch))
        self.d_loss_log.append(sum(self.d_loss_log_epoch) / len(self.d_loss_log_epoch))
        self.g_loss_log_epoch = []
        self.d_loss_log_epoch = []
        print(f"Epoche {epoch} done.")

    def conv_signal(self, signal_tensor, kernel_size=3):
        batch_size, signal_values, _ = signal_tensor.size()
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel = (torch.ones(1, 1, kernel_size) // kernel_size).to(signal_tensor.device)

        # Flattening des Signals für die Faltung
        flattened_signals = signal_tensor.permute(0, 2, 1).contiguous().view(batch_size * 2, 1, signal_values)

        # Anwendung der Faltung auf beide Koordinaten gleichzeitig
        conv_signal = nn.functional.conv1d(flattened_signals, kernel, padding=kernel_size // 2).view(batch_size, 2,
                                                                                              signal_values)
        return conv_signal.permute(0, 2, 1).contiguous()
    '''
    def train_RGAN(self, dataloader):
        #plot 1 real example
        data, _ = next(dataloader)
        self.plot_imgs(data,'real_example')
        for epoch in range(self.epochs):
            for batch in dataloader:
                input_data, labels = batch
                input_data, labels = input_data.to(self.device), labels.to(self.device)

                self.z = torch.randn(input_data.shape[0], input_data.shape[1], self.noise_size).to(self.device)
                input_data = input_data.to(dtype=self.z.dtype)# ,dtype=input_data.dtype).to(self.device)
                # Trainiere den Discriminator ncritic mal
                for _ in range(self.ncritic):
                    self.train_discriminator(input_data)
                # Trainiere den Generator
                self.train_generator()
            self.on_epoch_end(epoch)

            # Speichere Checkpoints
            if epoch % self.mlflow_interval == 0:
                self.plot_imgs(self.generator(self.z).detach(), f"epoch={epoch}", compare_data=input_data)
                self.save_checkpoint(epoch)
    '''

    def train_RCGAN(self, dataloader):
        for epoch in range(self.epochs):
            i=0
            for batch in dataloader:
                while i<5:
                    input_data, labels = batch
                    input_data, labels = input_data.to(self.device), labels.to(self.device)
                    input_data = input_data.to(dtype=torch.float32)# ,dtype=input_data.dtype).to(self.device)
                    # Train discriminator
                    for _ in range(self.ncritic):
                        z = self.sample_z(input_data)
                        z_labels = dataloader.rand_batch_labels(labels.shape[0]).to(self.device)
                        self.train_RCDiscriminator(input_data, labels, z, z_labels)
                    # Train Generator
                    z = self.sample_z(input_data)
                    z_labels = dataloader.rand_batch_labels(labels.shape[0]).to(self.device)
                    self.train_RCgenerator(z, z_labels)
                    i+=1
            self.on_epoch_end(epoch)

            # Speichere Checkpoints
            if epoch % self.mlflow_interval == 0:
                sigma = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10, 50,
                         100]
                sigma, best_mmd2 = self.perform_sigma_grid_search(epoch, sigma,
                                                                  fake_labels=dataloader.rand_batch_labels(self.vali_set.shape[0]).to(self.device))
                self.fft_analysis_plot(real=input_data, fake=self.generator(z, z_labels).detach(), plot=False)
                print(f"Epoche {epoch} done. \nSigma = {sigma}, mmd2={best_mmd2}.")
                self.plot_imgs(self.generator(z, z_labels).detach(), f"epoch={epoch}", compare_data=input_data, compare_labels=labels, z_labels=z_labels)#TODO: WIEDER RÜCKGÄNGIG MACHEN!!!self.plot_imgs(self.generator(self.z, labels).detach(), f"epoch={epoch}")
                #self.tsne_pca(epoch, fake_data=self.generator(self.z, self.z_labels).detach(), real_data=input_data)
                self.save_checkpoint(epoch)
    def sample_z(self, input_data):
        if self.scale:
            z = self.random_tensor((input_data.shape[0], input_data.shape[1], self.noise_size)).to(self.device) * self.scale
            #z = torch.randn(input_data.shape[0], input_data.shape[1], self.noise_size).to(self.device) * self.scale
        else:
            z = torch.full([input_data.shape[0], input_data.shape[1], self.noise_size], 0).to(dtype=torch.float32).to(
                self.device)
            random_values = (torch.rand([input_data.shape[0], 1, self.noise_size]) * 0.4 - 0.2).to(self.device)
            z += random_values
        return z

    def random_tensor(self, shape):
        random_samples = torch.rand(shape)
        random_tensor = 2*random_samples -1
        return random_tensor
    def adversarial_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_hat, y)

    def fft_analysis_plot(self, real, fake, plot=False):
        real = real.cpu().numpy()
        fake = fake.cpu().numpy()

        # Extrahiere die x- und y-Koordinaten für beide Signale
        real_x = real[0, :, 0]
        real_y = real[0, :, 1]
        fake_x = fake[0, :, 0]
        fake_y = fake[0, :, 1]

        # Annahme: signal_1 und signal_2 sind zwei Signale, die verglichen werden sollen
        signal_1 = real_x + 1j * real_y
        signal_2 = fake_x + 1j * fake_y
        # Anwenden eines Hamming-Fensters
        hamming_window = np.hamming(len(signal_1))
        signal_1 *= hamming_window
        signal_2 *= hamming_window

        # Fourier-Transformation
        fft_signal_1 = fft(signal_1)
        fft_signal_2 = fft(signal_2)

        # Berechne die Ähnlichkeit im Frequenzraum, z.B. durch die Kreuzkorrelation
        similarity = np.corrcoef(np.abs(fft_signal_1), np.abs(fft_signal_2))[0, 1]
        self.fft_similarity.append(similarity)
        if plot:
            # Plot der Zeitbereiche
            plt.figure(figsize=(12, 4))

            plt.subplot(2, 2, 1)
            plt.plot(real_x, label='Signal 1 (x)')
            plt.plot(real_y, label='Signal 1 (y)')
            plt.title('Zeitbereich - Signal 1')
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(fake_x, label='fake (x)')
            plt.plot(fake_y, label='fake (y)')
            plt.title('Zeitbereich - fake')
            plt.legend()

            # Plot der Frequenzbereiche
            plt.subplot(2, 2, 3)
            plt.plot(np.abs(fft_signal_1), label='real')
            plt.title('Frequenzbereich - real')

            plt.subplot(2, 2, 4)
            plt.plot(np.abs(fft_signal_2), label='fake')
            plt.title('Frequenzbereich - fake')

            plt.tight_layout()
            plt.savefig(r"C:\Users\uvuik\Desktop\FFT\Fig1.jpeg")
            plt.show()

        # Ausgabe der Ähnlichkeit
        print(f"Ähnlichkeit im Frequenzraum: {similarity}")

    def train_generator(self):
        self.optimizer_g.zero_grad()
        #z = torch.randn(input_data.shape[0], self.noise_size).to(self.device)
        fake_data = self.generator(self.z)
        #y_alt = torch.ones(input_data.shape[0], 2).to(self.device)
        y = torch.ones_like(self.discriminator(fake_data)).to(self.device)
        g_loss = self.adversarial_loss(self.discriminator(fake_data), y) #Aus Paper: -CE(RNNd(RNNg(Zn)),1) = -CE(RNNd(fake_data),1)
        g_loss.backward()
        self.optimizer_g.step()
        self.g_loss_log_epoch.append(g_loss.item())

    def train_RCgenerator(self, z, z_labels):
        self.optimizer_g.zero_grad()
        #fake_data = self.generator(self.z, labels)
        fake_data = self.generator(z, z_labels)
        # Calculating Gradient
        # conv_fake = self.conv_signal(fake_data, kernel_size=self.conv_window)
        # gradients_fake = torch.diff(conv_fake, dim=1)
        # gradients_fake = torch.cat([torch.zeros_like(gradients_fake[:, :1, :]), gradients_fake], dim=1)
        # cond_input_fake = torch.concat([self.z_labels.unsqueeze(-1), gradients_fake], dim=2)

        # Gloss(Z) = Dloss(RNNg(Z),1) = -CE(RNNd(RNNg(Z)),1)
        #TODO: Wieder ändern wenn kein label embedding
        d_output = self.discriminator(fake_data, z_labels)#self.z_labels)  # cond_input_fake) #TODO: WIRD HIER DER DISKRIMINATOR TRAINIERT?
        y = torch.ones_like(d_output).to(self.device)
        g_loss = self.adversarial_loss(d_output, y) #Aus Paper: -CE(RNNd(RNNg(Zn)),1) = -CE(RNNd(fake_data),1)
        g_loss.backward()
        self.optimizer_g.step()
        self.g_loss_log_epoch.append(g_loss.item())

    # Train discriminator: Maximize log(D(x) + log(1-D(G(z)))
    def train_discriminator(self, input_data):
        self.optimizer_d.zero_grad()

        # How well can it label as real
        y_hat_real = self.discriminator(input_data)
        y_real = torch.ones(y_hat_real.shape, dtype=y_hat_real.dtype).to(self.device)
        real_loss = self.adversarial_loss(y_hat_real,y_real) # Target size (torch.Size([32, 2])) must be the same as input size (torch.Size([32, 5760, 1]))

        # How well can it label generated labels as fake
        y_hat_fake = self.discriminator(self.generator(self.z).detach())
        y_fake = torch.zeros(y_hat_fake.shape).to(self.device)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

        d_loss = (real_loss + fake_loss) / 2 #TODO: WIESO WIRD HIER HALBIERT?
        #d_loss = (real_loss + fake_loss)  # TODO: Nicht mehr halbiert
        #log_dict = {'d_loss': d_loss.item()}
        d_loss.backward()
        self.optimizer_d.step()
        self.d_loss_log_epoch.append(d_loss.item())

    def train_RCDiscriminator(self, input_data, labels,z, z_labels):
        self.optimizer_d.zero_grad()

        # Convolution and calcuating gradient---------------------------------------------------------------------------
        # Smoothing the original signal -> convolution with window_size, and mean-kernel
        #conv_real = self.conv_signal(input_data, kernel_size=self.conv_window)
        #Gradient-Tensor berechnen
        #gradients_real = torch.diff(conv_real, dim=1)
        #gradients_real = torch.cat([torch.zeros_like(gradients_real[:, :1, :]), gradients_real], dim=1)
        #cond_input_real = torch.concat([labels, gradients_real], dim=2)
        #---------------------------------------------------------------------------------------------------------------

        # How well can it label as real
        #Gradient als weiteren input. TODO: Falls gewünscht wieder ändern!
        y_hat_real = self.discriminator(input_data, labels)  # ,cond_input)
        y_real = torch.ones(y_hat_real.shape, dtype=y_hat_real.dtype).to(self.device)
        real_loss = self.adversarial_loss(y_hat_real,y_real) # Target size (torch.Size([32, 2])) must be the same as input size (torch.Size([32, 5760, 1]))

        # How well can it label generated labels as fake
        #--------------------------------------------------------------------------------------------------------------
        # Im Originalen Code werden hier keine labels generiert, sondern die lables des batches verwendet
        #gen_output = self.generator(self.z, labels).detach()
        #y_hat_fake = self.discriminator(gen_output, labels)
        #y_fake = torch.zeros(y_hat_fake.shape).to(self.device)
        #fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
        # --------------------------------------------------------------------------------------------------------------

        #----------------Testing generating labels----------------------------------------------------------------------
        #Sample z
        gen_output = self.generator(z, z_labels).detach()
        #conv_fake = self.conv_signal(gen_output, kernel_size=self.conv_window)
        #gradients_fake = torch.diff(conv_fake, dim=1)
        #gradients_fake = torch.cat([torch.zeros_like(gradients_fake[:, :1, :]), gradients_fake], dim=1)

        #cond_input_fake = torch.concat([self.z_labels.unsqueeze(-1), gradients_fake], dim=2)
        y_hat_fake = self.discriminator(gen_output, z_labels)
        y_fake = torch.zeros(y_hat_fake.shape).to(self.device)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
        #---------------------------------------------------------------------------------------------------------------

        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optimizer_d.step()
        self.d_loss_log_epoch.append(d_loss.item())

    def perform_sigma_grid_search(self, epoch, sigma_values, fake_labels=None):
        best_sigma = None
        best_mmd2 = 1000
        real_sample = self.vali_set
        noise = torch.randn(real_sample.shape[0], real_sample.shape[1], self.noise_size).to(self.device)
        if str(self.GANtype) == "RGAN":
            fake_sample = self.generator(noise).detach()
        elif str(self.GANtype) == "RCGAN":
            fake_sample = self.generator(noise, fake_labels).detach()
        # real_sample_x = real_sample[:, :, 0].squeeze()
        # real_sample_y = real_sample[:, :, 1].squeeze()
        # fake_sample_x = fake_sample[:, :, 0].squeeze()
        # fake_sample_y = fake_sample[:, :, 1].squeeze()

        for sigma in sigma_values:
            current_mmd2, _ = mmd.mix_rbf_mmd2_and_ratio(real_sample, fake_sample, sigma)
            '''
            current_mmd2_x, _ = mmd.mix_rbf_mmd2_and_ratio(real_sample_x, fake_sample_x, sigma)
            current_mmd2_y, _ = mmd.mix_rbf_mmd2_and_ratio(real_sample_y, fake_sample_y, sigma)
            current_mmd2 = (current_mmd2_x + current_mmd2_y) / 2
            '''
            if current_mmd2 < best_mmd2 and current_mmd2 > 0:
                best_mmd2 = current_mmd2
                best_sigma = sigma
                if epoch > 10 and best_mmd2 < self.best_mmd2:
                    self.best_mmd2 = best_mmd2
                    self.best_epoch = epoch

        self.mmd2.append(float(best_mmd2))
        return best_sigma, best_mmd2

    def save_checkpoint(self, epoch):
        checkpoint = {
            'GANtype': self.GANtype,
            'GANargs': self.models['generator']['args'],
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'g_loss_log': self.g_loss_log,
            'd_loss_log': self.d_loss_log,
            'mmd2': self.mmd2,
            'fft_similarity': self.fft_similarity,
            'best_epoch': self.best_epoch
        }

        # Speichere das Dictionary in einer Datei
        filename = fr'{self.savepath}\Evaluation_{epoch}.pth'
        torch.save(checkpoint, filename)
        self.plot_losses_and_mmd2(epoch)
    def tsne_pca(self, epoch, real_data = None, fake_data=None):
        real_data = real_data[:,:,0].squeeze().cpu()#real_data.reshape((real_data.shape[0],-1)).cpu()
        fake_data = fake_data[:,:,0].squeeze().cpu()#fake_data.reshape((fake_data.shape[0],-1)).cpu()

        pca = PCA(n_components=2)
        real_data_pca = pca.fit_transform(real_data)
        fake_data_pca = pca.transform(fake_data)

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity = 25)
        real_data_tsne = tsne.fit_transform(real_data)
        fake_data_tsne = tsne.fit_transform(fake_data)

        # Plot results
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.scatter(real_data_pca[:, 0], real_data_pca[:, 1], label='Real Data')
        plt.scatter(fake_data_pca[:, 0], fake_data_pca[:, 1], label='Fake Data', alpha=0.5)
        plt.title('PCA - Real vs Fake Data')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.scatter(real_data_tsne[:, 0], real_data_tsne[:, 1], label='Real Data')
        plt.scatter(fake_data_tsne[:, 0], fake_data_tsne[:, 1], label='Fake Data', alpha=0.5)
        plt.title('t-SNE - Real vs Fake Data')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.scatter(real_data_pca[:, 0], real_data_pca[:, 1], label='Real Data')
        plt.title('PCA - Real Data Only')

        plt.subplot(2, 2, 4)
        plt.scatter(fake_data_pca[:, 0], fake_data_pca[:, 1], label='Fake Data', alpha=0.5)
        plt.title('PCA - Fake Data Only')

        plt.tight_layout()
        plt.savefig(f'{self.savepath}\PCA_TSNE_epoch{epoch}.jpg')
        plt.close()

    def plot_imgs(self, data, filename, compare_data=None, compare_labels = None, z_labels = None):
        frequency = self.frequency
        time_axis = np.arange(0, data.shape[1] / frequency, 1 / frequency)
        data = data.cpu().numpy()

        if compare_data is not None and compare_labels is not None:
            compare_data = compare_data.cpu().numpy()
            fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 12))
            micsac3_ = compare_labels[0,:].squeeze().cpu().numpy()
            micsac3 = np.array([1 if (micsac3_[max(0, i - 1)] + micsac3_[min(len(micsac3_) - 1, i + 1)]) % 2 == 1 and micsac3_[
                i] == 1 else 0 for i in range(len(micsac3_))])
            micsac4_ = compare_labels[1,:].squeeze().cpu().numpy()
            micsac4 = np.array([1 if (micsac4_[max(0, i - 1)] + micsac4_[min(len(micsac4_) - 1, i + 1)]) % 2 == 1 and micsac4_[
                i] == 1 else 0 for i in range(len(micsac4_))])
            x3 = compare_data[0, :, 0]
            y3 = compare_data[0, :, 1]
            x4 = compare_data[1, :, 0]
            y4 = compare_data[1, :, 1]

            # Dritter Plot (oben rechts)
            ax3.plot(time_axis, x3, label='X')
            ax3.plot(time_axis, y3, label='Y')
            ax3.scatter(time_axis[micsac3 == 1], x3[micsac3 == 1], marker='x', color='red', label='Mikrosakkade')
            ax3.scatter(time_axis[micsac3 == 1], y3[micsac3 == 1], marker='x', color='red')
            ax3.set_title('real', fontsize=24)
            ax3.legend(fontsize=20)

            # Vierter Plot (unten rechts)
            ax4.plot(time_axis, x4, label='X')
            ax4.plot(time_axis, y4, label='Y')
            ax4.scatter(time_axis[micsac4 == 1], x4[micsac4 == 1], marker='x', color='red', label='Mikrosakkade')
            ax4.scatter(time_axis[micsac4 == 1], y4[micsac4 == 1], marker='x', color='red')
            ax4.set_title('real', fontsize=24)
            ax4.set_xlabel('Zeit in Sekunden [s]', fontsize=22)
            ax3.set_xticks(np.arange(0, compare_data.shape[1] / frequency + 1, 1))
            ax4.set_xticks(np.arange(0, compare_data.shape[1] / frequency + 1, 1))
            ax4.set_xticks(ax4.get_xticks())
            ax4.set_xticklabels(ax4.get_xticks(), fontsize=18)
            ax4.legend(fontsize=20)

        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.suptitle('Verlauf generierter Trajektorien (links) und \nrealer Trajektorien des GazeBase-Datensatzes (rechts)', fontsize=32)
        x1 = data[0, :, 0]
        x2 = data[1, :, 0]
        y1 = data[0, :, 1]
        y2 = data[1, :, 1]
        if str(self.GANtype) == "RCGAN":
            #Wenn RCGAN dann positionen der Mikrosakkaden plotten
            micsac1_ = z_labels[0, :].squeeze().cpu().numpy()
            micsac1 = np.array([1 if (micsac1_[max(0, i - 1)] + micsac1_[min(len(micsac1_) - 1, i + 1)]) % 2 == 1 and micsac1_[i] == 1 else 0 for i in range(len(micsac1_))])
            micsac2_ = z_labels[1,:].squeeze().cpu().numpy()
            micsac2 = np.array([1 if (micsac2_[max(0, i - 1)] + micsac2_[min(len(micsac2_) - 1, i + 1)]) % 2 == 1 and micsac2_[
                i] == 1 else 0 for i in range(len(micsac2_))])
        else:
            micsac1, micsac2 = None, None


        # Erster Plot (oben links oder einziger Plot oben)
        ax1.plot(time_axis, x1, label='X')
        ax1.plot(time_axis, y1, label='Y')
        if micsac1 is not None:
            ax1.scatter(time_axis[micsac1 == 1], x1[micsac1 == 1], marker='x', color='red', label='Mikrosakkade')
            ax1.scatter(time_axis[micsac1 == 1], y1[micsac1 == 1], marker='x', color='red')
        ax1.set_ylabel('Position (dimensionslos)', fontsize=22)
        ax1.set_title('synthetisch', fontsize=24)
        ax1.legend(fontsize=20)

        # Zweiter Plot (unten links oder einziger Plot unten)
        ax2.plot(time_axis, x2, label='X')
        ax2.plot(time_axis, y2, label='Y')
        if micsac2 is not None:
            ax2.scatter(time_axis[micsac2 == 1], x2[micsac2 == 1], marker='x', color='red', label='Mikrosakkade')
            ax2.scatter(time_axis[micsac2 == 1], y2[micsac2 == 1], marker='x', color='red')
        ax2.set_ylabel('Position (dimensionslos)', fontsize=22)
        ax2.set_xlabel('Zeit in Sekunden [s]', fontsize=22)
        ax2.set_title('synthetisch', fontsize=24)
        ax1.set_xticks(np.arange(0, data.shape[1] / frequency + 1, 1))
        ax2.set_xticks(np.arange(0, data.shape[1] / frequency + 1, 1))
        ax2.legend(fontsize=18)
        ax2.set_yticks(ax2.get_yticks())
        ax2.set_yticklabels(ax2.get_yticks(), fontsize=18)
        ax2.set_xticks(ax2.get_xticks())
        ax2.set_xticklabels(ax2.get_xticks(), fontsize=18)
        ax1.set_yticks(ax2.get_yticks())
        ax1.set_yticklabels(ax1.get_yticks(),fontsize=18)
        ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
        ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
        fig.tight_layout()

        # Speichern der Figur
        fig.savefig(fr'{self.savepath}\{filename}.png')
        plt.close()
        return True

    def plot_losses_and_mmd2(self, epoch):
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

        # Subplot für Discriminator und Generator Loss
        ax1.plot(np.linspace(0, epoch, len(self.d_loss_log)), self.d_loss_log, label='Discriminator Loss', color='blue')
        ax2 = ax1.twinx()  # Zweite Y-Achse für den rechten Plot
        ax2.plot(np.linspace(0, epoch, len(self.g_loss_log)), self.g_loss_log, label='Generator Loss', color='red')

        ax1.set_ylabel('Discriminator Loss', color='blue')
        ax2.set_ylabel('Generator Loss', color='red')

        ax1.tick_params('y', colors='blue')
        ax2.tick_params('y', colors='red')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax1.grid(True)
        ax2.grid(True)
        min_y = 0
        max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(min_y, max_y)
        ax2.set_ylim(min_y, max_y)


        # Subplot für MMD2 (zweite Zeile des Plots)
        ax3.plot(np.linspace(0, epoch, len(self.mmd2)), self.mmd2, label='MMD2', color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MMD2', color='green')
        ax3.tick_params('y', colors='green')
        ax3.grid(True)

        # Layout verbessern
        plt.suptitle('Discriminator and Generator Loss, MMD2 Over Time')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
        plt.savefig(fr'{self.savepath}\Evaluation.png')
        plt.close()


logger = make_logger(__file__)

