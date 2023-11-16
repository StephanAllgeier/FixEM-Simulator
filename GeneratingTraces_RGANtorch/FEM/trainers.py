"""
References:
    - https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
"""
import torch
import numpy as np
import torch.nn as nn
import torchgan
from torch.nn import BCELoss, BCEWithLogitsLoss
from GeneratingTraces_RGANtorch.FEM import make_logger
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
import GeneratingTraces_RGANtorch.FEM.mmd as mmd
import mlflow
import matplotlib.pyplot as plt
logger = make_logger(__file__)
from skopt import gp_minimize
from skopt.space import Real
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.fft import fft

class SequenceTrainer:
    def __init__(self, models, recon, ncritic, losses_list, epochs, retain_checkpoints, checkpoints, mlflow_interval, device, noise_size, vali_set, savepath, GANtype, eval_frequency=1, conv_window = 10, scale=None):
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
        #self.perform_sigma_optimization(epoch, fake_labels=fake_labels)
        #self.evaluate_mmd2(epoch)
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

    def train_RCGAN(self, dataloader):
        #plot 1 real example
        #data, _ = next(dataloader)
        #self.plot_imgs(data,'real_example')
        for epoch in range(self.epochs):
            for batch in dataloader:

                input_data, labels = batch
                input_data, labels = input_data.to(self.device), labels.to(self.device)
                if self.scale:
                    self.z = torch.randn(input_data.shape[0], input_data.shape[1], self.noise_size).to(self.device)*self.scale
                else:
                    self.z = torch.full([input_data.shape[0], input_data.shape[1], self.noise_size],0).to(dtype=torch.float32).to(self.device)##
                self.z_labels = dataloader.rand_batch_labels(labels.shape[0]).to(self.device)
                input_data = input_data.to(dtype=self.z.dtype)# ,dtype=input_data.dtype).to(self.device)
                # Train discriminator
                for _ in range(self.ncritic):
                    self.train_RCgenerator(labels)
                self.train_RCDiscriminator(input_data, labels)
                # Trainiere den Generator n_critic mal

            self.on_epoch_end(epoch)

            # Speichere Checkpoints
            if epoch % self.mlflow_interval == 0:
                sigma = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10, 50, 100]
                sigma, best_mmd2 = self.perform_sigma_grid_search(epoch, sigma, fake_labels=dataloader.rand_batch_labels(self.vali_set.shape[0]).to(self.device))
                self.fft_analysis_plot(real=input_data, fake=self.generator(self.z, self.z_labels).detach(), plot=False)
                print(f"Epoche {epoch} done. \nSigma = {sigma}, mmd2={best_mmd2}.")
                self.plot_imgs(self.generator(self.z, self.z_labels).detach(), f"epoch={epoch}", compare_data=input_data, compare_labels=labels)#TODO: WIEDER RÜCKGÄNGIG MACHEN!!!self.plot_imgs(self.generator(self.z, labels).detach(), f"epoch={epoch}")
                #self.tsne_pca(epoch, fake_data=self.generator(self.z, self.z_labels).detach(), real_data=input_data)
                self.save_checkpoint(epoch)
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
        self.g_loss_log.append(g_loss.item())

    def train_RCgenerator(self, labels):
        self.optimizer_g.zero_grad()
        #fake_data = self.generator(self.z, labels)
        fake_data = self.generator(self.z, self.z_labels)
        # Calculating Gradient
        # conv_fake = self.conv_signal(fake_data, kernel_size=self.conv_window)
        # gradients_fake = torch.diff(conv_fake, dim=1)
        # gradients_fake = torch.cat([torch.zeros_like(gradients_fake[:, :1, :]), gradients_fake], dim=1)
        # cond_input_fake = torch.concat([self.z_labels.unsqueeze(-1), gradients_fake], dim=2)

        # Gloss(Z) = Dloss(RNNg(Z),1) = -CE(RNNd(RNNg(Z)),1)
        #TODO: Wieder ändern wenn kein label embedding
        d_output = self.discriminator(fake_data,self.z_labels)  # cond_input_fake)
        y = torch.ones_like(d_output).to(self.device)
        g_loss = self.adversarial_loss(d_output, y) #Aus Paper: -CE(RNNd(RNNg(Zn)),1) = -CE(RNNd(fake_data),1)
        g_loss.backward()
        self.optimizer_g.step()
        self.g_loss_log.append(g_loss.item())

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
        self.d_loss_log.append(d_loss.item())

    def train_RCDiscriminator(self, input_data, labels):
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
        gen_output = self.generator(self.z, labels).detach()
        #gen_output = self.generator(self.z, self.z_labels).detach()
        conv_fake = self.conv_signal(gen_output, kernel_size=self.conv_window)
        gradients_fake = torch.diff(conv_fake, dim=1)
        gradients_fake = torch.cat([torch.zeros_like(gradients_fake[:, :1, :]), gradients_fake], dim=1)
        #TODO: Wenn Gradienten Gewünscht, dann wieder Ändern
        #cond_input_fake = torch.concat([self.z_labels.unsqueeze(-1), gradients_fake], dim=2)
        y_hat_fake = self.discriminator(gen_output, self.z_labels)
        y_fake = torch.zeros(y_hat_fake.shape).to(self.device)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
        #---------------------------------------------------------------------------------------------------------------

        d_loss = (real_loss + fake_loss) / 2
        #d_loss = (real_loss + fake_loss)
        d_loss.backward()
        self.optimizer_d.step()
        self.d_loss_log.append(d_loss.item())

    def objective(self, sigma):
        current_mmd2_x, _ = mmd.mix_rbf_mmd2_and_ratio(self.real_sample_x, self.fake_sample_x, sigma[0])
        current_mmd2_y, _ = mmd.mix_rbf_mmd2_and_ratio(self.real_sample_y, self.fake_sample_y, sigma[0])
        current_mmd2 = (current_mmd2_x + current_mmd2_y) / 2
        return current_mmd2

    def perform_sigma_optimization(self, epoch, fake_labels=None):
        self.real_sample = self.vali_set
        noise = torch.randn(self.real_sample.shape[0], self.real_sample.shape[1], self.noise_size).to(self.device)
        if str(self.GANtype) == "RGAN":
            self.fake_sample = self.generator(noise).detach()
        elif str(self.GANtype) == "RCGAN":
            self.fake_sample = self.generator(noise, fake_labels).detach()
        min_len = min(len(self.real_sample), len(self.fake_sample))
        selected_indices = np.random.choice(len(self.real_sample), min_len, replace=False)
        self.real_sample = self.real_sample[selected_indices]
        self.real_sample_x = self.real_sample[:, :, 0].squeeze()
        self.real_sample_y = self.real_sample[:, :, 1].squeeze()
        self.fake_sample = self.fake_sample[:min_len]
        self.fake_sample_x = self.fake_sample[:, :, 0].squeeze()
        self.fake_sample_y = self.fake_sample[:, :, 1].squeeze()

        # Definiere den Raum der Sigma-Werte für die Optimierung
        space = Real(1e-3, 1e3, name='sigma', prior='log-uniform')

        # Führe die Bayesian Optimization durch
        result = gp_minimize(
            self.objective,
            dimensions=[space],
            n_calls=20,  # Anzahl der Evaluierungen
            random_state=42
        )

        best_sigma = result.x[0]
        best_mmd2 = result.fun
        self.mmd2.append(float(best_mmd2))
        if epoch > 10 and best_mmd2 < self.best_mmd2 and best_mmd2 > 0:
            self.best_mmd2 = best_mmd2
            self.best_epoch = epoch
        return best_sigma, best_mmd2

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

    def plot_imgs(self, data, filename, compare_data=None, compare_labels = None):
        frequency = 250
        time_axis = np.arange(0, data.shape[1] / frequency, 1 / frequency)
        data = data.cpu().numpy()

        if compare_data is not None and compare_labels is not None:
            compare_data = compare_data.cpu().numpy()
            fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
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
            ax3.set_ylabel('Vergleichsdaten 1')
            ax3.legend()

            # Vierter Plot (unten rechts)
            ax4.plot(time_axis, x4, label='X')
            ax4.plot(time_axis, y4, label='Y')
            ax4.scatter(time_axis[micsac4 == 1], x4[micsac4 == 1], marker='x', color='red', label='Mikrosakkade')
            ax4.scatter(time_axis[micsac4 == 1], y4[micsac4 == 1], marker='x', color='red')
            ax4.set_ylabel('Vergleichsdaten 2')
            ax4.set_xlabel('Zeit in Sekunden [s]')
            ax3.set_xticks(np.arange(0, compare_data.shape[1] / frequency + 1, 1))
            ax4.set_xticks(np.arange(0, compare_data.shape[1] / frequency + 1, 1))
            ax4.legend()

        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        x1 = data[0, :, 0]
        x2 = data[1, :, 0]
        y1 = data[0, :, 1]
        y2 = data[1, :, 1]
        if str(self.GANtype) == "RCGAN":
            #Wenn RCGAN dann positionen der Mikrosakkaden plotten
            micsac1_ = self.z_labels[0, :].squeeze().cpu().numpy()
            micsac1 = np.array([1 if (micsac1_[max(0, i - 1)] + micsac1_[min(len(micsac1_) - 1, i + 1)]) % 2 == 1 and micsac1_[i] == 1 else 0 for i in range(len(micsac1_))])
            micsac2_ = self.z_labels[1,:].squeeze().cpu().numpy()
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
        ax1.set_ylabel('Beispiel 1')
        ax1.legend()

        # Zweiter Plot (unten links oder einziger Plot unten)
        ax2.plot(time_axis, x2, label='X')
        ax2.plot(time_axis, y2, label='Y')
        if micsac2 is not None:
            ax2.scatter(time_axis[micsac2 == 1], x2[micsac2 == 1], marker='x', color='red', label='Mikrosakkade')
            ax2.scatter(time_axis[micsac2 == 1], y2[micsac2 == 1], marker='x', color='red')
        ax2.set_ylabel('Beispiel 2')
        ax2.set_xlabel('Zeit in Sekunden [s]')
        ax1.set_xticks(np.arange(0, data.shape[1] / frequency + 1, 1))
        ax2.set_xticks(np.arange(0, data.shape[1] / frequency + 1, 1))
        ax2.legend()

        fig.tight_layout()

        # Speichern der Figur
        fig.savefig(fr'{self.savepath}\{filename}.png')
        plt.close()
        return True

    def plot_losses_and_mmd2(self, epoch):
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

        # Subplot für Discriminator und Generator Loss
        ax1.plot(np.linspace(0, epoch + 1, len(self.d_loss_log)), self.d_loss_log, label='Discriminator Loss', color='blue')
        ax2 = ax1.twinx()  # Zweite Y-Achse für den rechten Plot
        ax2.plot(np.linspace(0, epoch + 1, len(self.g_loss_log)), self.g_loss_log, label='Generator Loss', color='red')

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
        ax3.plot(np.linspace(0, epoch + 1, len(self.mmd2)), self.mmd2, label='MMD2', color='green')
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


class BinaryClassificationTrainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 sampler_train=None,
                 sampler_test=None,
                 log_to_mlflow=True,
                 loss_function=BCEWithLogitsLoss(),
                 metrics_prepend=''):
        self.optimizer = optimizer
        self.sampler_train = sampler_train
        self.sampler_test = sampler_test
        self.loss_function = loss_function
        self.model = model
        self.log_to_mlflow = log_to_mlflow
        self.tiled = sampler_train.tile
        self.metrics_prepend = metrics_prepend

    def train(self, epochs, evaluate_interval=100):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            X_train, y_train = self.sampler_train.sample()

            logits_train = self.model(X_train)
            loss = self.loss_function(logits_train, y_train)
            loss.backward()
            self.optimizer.step()

            if epoch % evaluate_interval == 0:
                with torch.no_grad():
                    X_test, y_test = self.sampler_test.sample()
                    logger.debug(f'[Train sizes {X_train.shape} {y_train.shape}]')
                    logger.debug(f'[Test sizes {X_test.shape} {y_test.shape}]')
                    metrics = self.evaluate(X_test, y_test,
                                            X_train, y_train)
                    msg = f'[epoch {epoch}]'
                    msg += ''.join(f'[{m} {np.round(v,4)}]'
                                   for m, v in metrics.items()
                                   if m.endswith('balanced_accuracy') or
                                   m.endswith('matheus'))
                    logger.info(msg)
                    if self.log_to_mlflow:
                        mlflow.log_metrics(metrics, step=epoch)

    def evaluate(self, X_test, y_test, X_train, y_train):
        def _calculate(X, y, name):
            logits = self.model(X)
            probs = torch.sigmoid(logits)
            y_pred = probs.round()
            y_true = y
            return self.calculate_metrics(y_true, y_pred, logits, probs, name)

        mp = self.metrics_prepend
        return {**_calculate(X_test, y_test, f'{mp}test'),
                **_calculate(X_train, y_train, f'{mp}train')}

    def calculate_metrics(self, y_true, y_pred, logits, probs, name=''):
        y_true_ = (y_true[:, 0] if self.tiled else y_true).cpu()
        y_pred_ = (y_pred.mode().values if self.tiled else y_pred).cpu()

        mask_0 = (y_true_ == 0)
        mask_1 = (y_true_ == 1)

        hits = (y_true_ == y_pred_).float()
        bas = balanced_accuracy_score(y_true_, y_pred_)
        matthews = matthews_corrcoef(y_true_, y_pred_)

        return {f'{name}_accuracy': hits.mean().item(),
                f'{name}_balanced_accuracy': bas,
                f'{name}_accuracy_0': hits[mask_0].mean().item(),
                f'{name}_accuracy_1': hits[mask_1].mean().item(),
                f'{name}_loss': self.loss_function(logits, y_true).item(),
                f'{name}_loss_0': self.loss_function(logits[mask_0],
                                                     y_true[mask_0]).item(),
                f'{name}_loss_1': self.loss_function(logits[mask_1],
                                                     y_true[mask_1]).item(),
                f'{name}_matthews': matthews}

