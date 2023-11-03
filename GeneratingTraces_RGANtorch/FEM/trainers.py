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

class SequenceTrainer:
    def __init__(self, models, recon, ncritic, losses_list, epochs, retain_checkpoints, checkpoints, mlflow_interval, device, noise_size, vali_set, savepath, eval_frequency=1):
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
        self.best_mmd2 = 1000
        self.vali_set = vali_set
        self.eval_frequency = eval_frequency
        self.best_epoch = 0
        self.savepath = savepath

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

    def on_epoch_end(self, epoch):
        sigma = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        #self.perform_sigma_optimization(epoch)
        #self.perform_sigma_grid_search(epoch, sigma)
        #self.evaluate_mmd2(epoch)
        print(f"{epoch} done")

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
                self.plot_imgs(self.generator(self.z).detach(), f"epoch={epoch}")
                self.save_checkpoint(epoch)

    def train_RCGAN(self, dataloader):
        #plot 1 real example
        data, _ = next(dataloader)
        self.plot_imgs(data,'real_example')
        for epoch in range(self.epochs):
            for batch in dataloader:

                input_data, labels = batch
                input_data, labels = input_data.to(self.device), labels.to(self.device)

                self.z = torch.randn(input_data.shape[0], input_data.shape[1], self.noise_size).to(self.device)
                #self.z_labels = torch.randn(input_data.shape[0], input_data.shape[1], labels.shape[-1]).to(self.device)
                input_data = input_data.to(dtype=self.z.dtype)# ,dtype=input_data.dtype).to(self.device)
                # Trainiere den Discriminator ncritic mal
                for _ in range(self.ncritic):
                    self.train_RCdiscriminator(input_data, labels)
                # Trainiere den Generator
                self.train_RCgenerator(labels)
            self.on_epoch_end(epoch)

            # Speichere Checkpoints
            if epoch % self.mlflow_interval == 0:
                self.plot_imgs(self.generator(self.z, labels).detach(), f"epoch={epoch}")
                self.save_checkpoint(epoch)
    def adversarial_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_hat, y)

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
        fake_data = self.generator(self.z, labels)

        # Gloss(Z) = Dloss(RNNg(Z),1) = -CE(RNNg(Z),1)
        d_output = self.discriminator(fake_data, labels)
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
        y_hat_fake = self.discriminator(self.generator(self.z).detach()) #Todo: Use the same ones as processed in generator step, does this work?
        y_fake = torch.zeros(y_hat_fake.shape).to(self.device)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

        d_loss = (real_loss + fake_loss) / 2 #TODO: WIESO WIRD HIER HALBIERT?
        #log_dict = {'d_loss': d_loss.item()}
        d_loss.backward()
        self.optimizer_d.step()
        self.d_loss_log.append(d_loss.item())
    def train_RCdiscriminator(self, input_data, labels):
        self.optimizer_d.zero_grad()
        # How well can it label as real
        y_hat_real = self.discriminator(input_data, labels)
        y_real = torch.ones(y_hat_real.shape, dtype=y_hat_real.dtype).to(self.device)
        real_loss = self.adversarial_loss(y_hat_real,y_real) # Target size (torch.Size([32, 2])) must be the same as input size (torch.Size([32, 5760, 1]))

        # How well can it label generated labels as fake
        #Im Originalen Code werden hier keine labels generiert, sondern die lables des batches verwendet
        gen_output = self.generator(self.z, labels).detach()
        #gen_output = self.generator(self.z, self.z_labels).detach() #TODO: Was sind hier die Labels dazu, die Generiert werden? Wie bekomme ich diese?

        y_hat_fake = self.discriminator(gen_output, labels) #Todo: Hier auch die generierten Labels???
        y_fake = torch.zeros(y_hat_fake.shape).to(self.device)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_d.step()
        self.d_loss_log.append(d_loss.item())

    def objective(self, sigma):
        current_mmd2_x, _ = mmd.mix_rbf_mmd2_and_ratio(self.real_sample_x, self.fake_sample_x, sigma[0])
        current_mmd2_y, _ = mmd.mix_rbf_mmd2_and_ratio(self.real_sample_y, self.fake_sample_y, sigma[0])
        current_mmd2 = (current_mmd2_x + current_mmd2_y) / 2
        return current_mmd2

    def perform_sigma_optimization(self, epoch):
        self.real_sample = self.vali_set
        self.fake_sample = self.generator(self.z).detach()
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

        if epoch > 10 and best_mmd2 < self.best_mmd2 and best_mmd2 > 0:
            self.best_mmd2 = best_mmd2
            self.best_epoch = epoch

        self.mmd2.append(float(best_mmd2))
        return best_sigma, best_mmd2

    def perform_sigma_grid_search(self, epoch, sigma_values):
        best_sigma = None
        best_mmd2 = 1000
        real_sample = self.vali_set
        fake_sample = self.generator(self.z).detach()
        min_len = 3  # min(len(real_sample), len(fake_sample))
        selected_indices = np.random.choice(len(real_sample), min_len, replace=False)
        real_sample = real_sample[selected_indices]
        real_sample_x = real_sample[:, :, 0].squeeze()
        real_sample_y = real_sample[:, :, 1].squeeze()
        fake_sample = fake_sample[:min_len]
        fake_sample_x = fake_sample[:, :, 0].squeeze()
        fake_sample_y = fake_sample[:, :, 1].squeeze()

        for sigma in sigma_values:
            current_mmd2_x, _ = mmd.mix_rbf_mmd2_and_ratio(real_sample_x, fake_sample_x, sigma)
            current_mmd2_y, _ = mmd.mix_rbf_mmd2_and_ratio(real_sample_y, fake_sample_y, sigma)
            current_mmd2 = (current_mmd2_x + current_mmd2_y) / 2

            if current_mmd2 < best_mmd2 and current_mmd2>0:
                best_mmd2 = current_mmd2
                best_sigma = sigma
                if epoch > 10:
                    self.best_mmd2 = best_mmd2
                    self.best_epoch = epoch

        self.mmd2.append(float(best_mmd2))
        return best_sigma, best_mmd2

    def save_model(self):


    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'g_loss_log': self.g_loss_log,
            'd_loss_log': self.d_loss_log,
            'mmd2': self.mmd2,
            'best_epoch': self.best_epoch
        }

        # Speichere das Dictionary in einer Datei
        filename = fr'{self.savepath}\Evaluation_{epoch}.pth'
        torch.save(checkpoint, filename)
        filename_txt = fr'{self.savepath}\Evaluation_{epoch}.txt'
        with open(filename_txt, 'w') as txt_file:
            for key, value in checkpoint.items():
                txt_file.write(f"{key}: {value}\n")
        self.plot_losses_and_mmd2(epoch)

    def plot_imgs(self, data, filename):
        frequency = 250 # 1000  TODO: ÄNDERN WENN f=1000
        time_axis = np.arange(0, data.shape[1] / frequency, 1 / frequency)
        data = data.cpu().numpy()
        # Erstelle die Plots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        x1 = data[0,:,0]
        x2 = data[1,:,0]
        y1 = data[0,:,1]
        y2 = data[1,:,1]
        # Erster Plot
        ax1.plot(time_axis, x1)
        ax1.plot(time_axis, x2)
        ax1.set_ylabel('x-Koordinate')

        # Zweiter Plot
        ax2.plot(time_axis, y1)
        ax2.plot(time_axis, y2)
        ax2.set_ylabel('y-Koordinate')
        ax2.set_xlabel('Zeit in Sekunden [s]')
        ax1.set_xticks(np.arange(0, data.shape[1] / frequency + 1, 1))
        ax2.set_xticks(np.arange(0, data.shape[1] / frequency + 1, 1))
        fig.tight_layout()
        # Speichere die Figur
        fig.savefig(fr'{self.savepath}\{filename}.png')
        plt.close()
        return True

    def plot_losses_and_mmd2(self, epoch):
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

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

