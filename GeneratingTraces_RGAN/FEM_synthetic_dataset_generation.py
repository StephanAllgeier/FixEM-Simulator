import data_utils
import pandas as pd
import numpy as np
import tensorflow as tf
import math, random, itertools
import pickle
import time
import json
import os
import math

import main
import plotting
from ProcessingData.Preprocessing import Augmentation
from main import get_constants
import tkinter as tk
from tkinter import filedialog

import model

def get_batch(samples, labels, batch_size, batch_idx):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    return samples[start_pos:end_pos], labels[start_pos:end_pos]

def load_dataset(folderpath, dataset_name, seq_length=3):
    csv_files = [f for f in os.listdir(folderpath) if f.endswith('.csv')]
    samples_x = []
    samples_y = []
    const_dict = main.get_constants(dataset_name)

    for i, file in enumerate(csv_files):
        filepath = os.path.join(folderpath, file)
        df = pd.read_csv(filepath)

        segments_x = Augmentation.Augmentation.slice_df(df[[const_dict['x_col']]], const_dict, seq_length)
        segments_y = Augmentation.Augmentation.slice_df(df[[const_dict['y_col']]], const_dict, seq_length)

        samples_x.extend(segments_x)
        samples_y.extend(segments_y)

    # Convert List to ndarray
    samples_x = np.array(samples_x)
    samples_y = np.array(samples_y)

    return samples_x, samples_y

def folder_input():
    # Ask the user for the folder path
    folder_path = input("Please enter the folder path: ")

    # Check if the entered path exists
    while not os.path.exists(folder_path):
        print("The entered folder path does not exist. Please check it again.")
        folder_path = input("Please enter the correct folder path: ")

    return folder_path

def run_training(folderpath, dataset_name, seq_len):
    if folderpath == None:
        folderpath  = input("Geben Sie den Pfad zum Eingabeordner ein: ")

    # Optional: Überprüfen, ob der eingegebene Pfad existiert
    if not os.path.exists(folderpath):
        print("Der eingegebene Pfad existiert nicht. Stellen Sie sicher, dass der Pfad korrekt ist.")
    else:
        # Nun kannst du input_folder verwenden, um auf den vom Benutzer festgelegten Eingabeordner zuzugreifen
        print(f"Eingabeordner festgelegt auf: {folderpath}")


    # Popup-Fenster für Ordnerauswahl öffnen
    #root = tk.Tk()
    #root.withdraw()
    #output_folder = filedialog.askdirectory(title="Wählen Sie einen Speicherordner aus")
    output_folder =r"C:\Users\uvuik\Desktop\Test"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Speicherort festgelegt auf: {output_folder}")

    # run experiment 5 times
    identifiers = ['FEM_synthetic_dataset' + str(i) for i  in range(1,6)]
    for identifier in identifiers:
        tf.compat.v1.reset_default_graph()
        print(f'loading data for {identifier}...')

        samples, labels = data_utils.FEM_task(folderpath, dataset_name, seq_len)

        #TODO: Herausfinden, wieso hier ein reshape vorgang duerchgeführt wird, eventuell überflüssig
        # zunächst mal nur die x-Koordinate testen, muss wieder entfernt werden
        samples['train'] = samples['train'][:, :, 0].reshape(samples['train'].shape[0], samples['train'].shape[1],1)
        samples['vali'] = samples['vali'][:, :, 0].reshape(samples['vali'].shape[0], samples['vali'].shape[1],1)
        samples['test'] = samples['test'][:, :, 0].reshape(samples['test'].shape[0], samples['test'].shape[1],1)


        train_seqs = samples['train']
        vali_seqs = samples['vali']
        test_seqs = samples['test']

        train_targets = labels['train'].reshape(-1,1)
        vali_targets = labels['vali'].reshape(-1,1)
        test_targets = labels['test'].reshape(-1,1)

        train_seqs, vali_seqs, test_seqs = data_utils.scale_data(train_seqs, vali_seqs, test_seqs) #Skaliert auf Werte zwischen -1 und 1
        print('Data has loaded...')

        #Training cofiguration
        lr = 0.1 #Learning-rate
        batch_size = 10
        num_epochs = 1000
        D_rounds = 1
        G_rounds = 3
        use_time = False

        seq_length = train_seqs.shape[1]
        num_generated_features = train_seqs.shape[2] # Eventuell muss hier 2 stehen, für x und y Koordinaten
        hidden_units_d = 100
        hidden_units_g = 100

        latent_dim=10 #dimension of the random latent space
        #TODO: aktuell keine cond_dim oder
        cond_dim = train_targets.shape[1] #SOLL == 1 sein, darf aber auch erstmal 0 bleiben
        # Deactivate eager execution
        tf.compat.v1.disable_eager_execution()
        CG = tf.compat.v1.placeholder(tf.float32, [batch_size, train_targets.shape[1]])
        CD = tf.compat.v1.placeholder(tf.float32, [batch_size, train_targets.shape[1]])
        Z = tf.compat.v1.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
        W_out_G = tf.Variable(tf.random.truncated_normal([hidden_units_g, num_generated_features]))
        b_out_G = tf.Variable(tf.random.truncated_normal([num_generated_features]))

        X = tf.compat.v1.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
        W_out_D = tf.Variable(tf.random.truncated_normal([hidden_units_d, 1]))
        b_out_D = tf.Variable(tf.random.truncated_normal([1]))

        def sample_Z(batch_size, seq_length, latent_dim, use_time=False, use_noisy_time=False):
            """
                Generates a random latent space sample for a given batch size, sequence length, and latent dimension.

                Parameters:
                - batch_size (int): Number of samples in the batch.
                - seq_length (int): Length of each sequence.
                - latent_dim (int): Dimensionality of the latent space.
                - use_time (bool): Flag indicating whether to include a time component in the generated sample.
                - use_noisy_time (bool): Flag indicating whether to add noise to the time component if `use_time` is True.

                Returns:
                np.ndarray: Random latent space sample with specified characteristics.
                """
            sample = np.random.normal(size=[batch_size, seq_length, latent_dim])
            if use_noisy_time or use_time:
                # time grid is time_grid_mult times larger than seq_length
                time_grid_mult = 5
                time_grid = (np.arange(seq_length * time_grid_mult) / ((seq_length * time_grid_mult) / 2)) - 1
                time_axes = []
                for i in range(batch_size):
                    # randomly chose a starting point in the time grid
                    starting_point = random.choice(np.arange(len(time_grid))[:-seq_length])
                    time_axis = time_grid[starting_point:starting_point + seq_length]
                    if use_noisy_time:
                        time_axis += np.random.normal(scale=2.0 / len(time_axis), size=len(time_axis))
                    time_axes.append(time_axis)
                sample[:, :, 0] = time_axes
            return sample

        def generator(z, c):
            """
                Generates a sequence using a conditional generative model with LSTM cells.

                Parameters:
                - z (tf.Tensor): Random seed for sequence generation.
                - c (tf.Tensor): Conditional embedding for guiding the generation process.

                Returns:
                tf.Tensor: Generated sequence based on the given random seed and conditional embedding.
                """
            with tf.compat.v1.variable_scope("generator") as scope:
                # each step of the generator takes a random seed + the conditional embedding
                repeated_encoding = tf.tile(c, [1, tf.shape(z)[1]])
                repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(z)[0], tf.shape(z)[1],
                                                                   cond_dim])
                generator_input = tf.concat([repeated_encoding, z], 2)

                cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_units_g, state_is_tuple=True)
                rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(
                    cell=cell,
                    dtype=tf.float32,
                    sequence_length=[seq_length] * batch_size,
                    inputs=generator_input)
                rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
                logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
                output_2d = tf.nn.tanh(logits_2d)
                output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
            return output_3d

        def discriminator(x, c, reuse=False):
            """
                Discriminates between real and generated sequences using an LSTM-based discriminator.

                Parameters:
                - x (tf.Tensor): Input sequence to be discriminated (real or generated).
                - c (tf.Tensor): Conditional embedding associated with the input sequence.
                - reuse (bool): Flag indicating whether to reuse variables in the discriminator.

                Returns:
                - tf.Tensor: Output probability (0 to 1) indicating the likelihood that the input sequence is real.
                - tf.Tensor: Logits representing the raw output of the discriminator before applying the sigmoid activation.
                """
            with tf.compat.v1.variable_scope("discriminator") as scope:
                # correct?
                if reuse:
                    scope.reuse_variables()

                # each step of the generator takes one time step of the signal to evaluate +
                # its conditional embedding
                repeated_encoding = tf.tile(c, [1, tf.shape(x)[1]])
                repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(x)[0], tf.shape(x)[1],
                                                                   cond_dim])
                decoder_input = tf.concat([repeated_encoding, x], 2)

                cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_units_d, state_is_tuple=True)
                rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(
                    cell=cell,
                    dtype=tf.float32,
                    inputs=decoder_input)
                rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units_g])
                logits = tf.matmul(rnn_outputs_flat, W_out_D) + b_out_D
                output = tf.nn.sigmoid(logits)
            return output, logits

        #---------------------------------------------------------------------------------------------------------------
        #Generative Adversarial Network (GAN) Training:

        #1. Generierung von Fake-Daten durch den Generator
        #2. Diskriminierung von Realen Daten
        #3. Diskriminierung von Fake-Daten (mit Wiederverwendung von Variablen)
        #4. Definition der Trainierbaren Variablen für Generator und Discriminator
        #5. Definition der Verlustfunktionen
        #6. Optimierung der Verlustfunktionen für Discriminator und Generator
        #7. Initialisierung der TensorFlow-Sitzung
        #--------------------------Aus experiment.py-----------------------------
        #num_signals = 1
        #cond_dim = 0
        #Z, X, CG, CD, CS = model.create_placeholders(batch_size, seq_length, latent_dim,
        #                                             num_signals, cond_dim) #TODO: nnum_signals = 1 wenn nur x und 2 wenn y

        G_sample = generator(Z, CG)
        D_real, D_logit_real = discriminator(X, CD)  # [0,1]
        D_fake, D_logit_fake = discriminator(G_sample, CG, reuse=True)  # [0,1]

        generator_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
        discriminator_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                                             labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                             labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                        labels=tf.ones_like(D_logit_fake)))

        D_solver = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr).minimize(D_loss,
                                                                                          var_list=discriminator_vars)
        G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=generator_vars)

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        #---------------------------------------------------------------------------------------------------------------

        #plot output from the same seed
        vis_z = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
        #TODO: Wieder ändern und mit Y_mb_vis anpassen
        X_mb_vis, Y_mb_vis = get_batch(train_seqs, train_targets, batch_size, 0)

        #X_mb_vis, Y_mb_vis = get_batch(train_seqs, train_targets, batch_size, 0)
        vis_sample = sess.run(G_sample, feed_dict={Z: vis_z, CG: Y_mb_vis})
        #TODO: ÄNDERN!!!!!
        plotting.vis_FEM_downsampled(vis_sample, seq_length, folder_to_save_to=r"C:\Users\uvuik\Desktop\Test",
                                               identifier=identifier, idx=0)

        # visualise some real samples
        vis_real = np.float32(vali_seqs[np.random.choice(len(vali_seqs), size=batch_size), :, :])
        #TODO: ÄNDERN!!!!!!
        plotting.vis_FEM_downsampled(vis_real, seq_length,folder_to_save_to=r"C:\Users\uvuik\Desktop\Test",
                                               identifier=identifier + '_real', idx=0)

        trace = open(output_folder + '/' + identifier + '.trace.txt', 'w')
        trace.write('epoch D_loss G_loss time\n')
        print('epoch\tD_loss\tG_loss\ttime\n')
        t0 = time.time()

        def train_generator(batch_idx, offset):
            # update the generator
            for g in range(G_rounds):
                X_mb, Y_mb = get_batch(train_seqs, train_targets, batch_size, batch_idx + g + offset)
                _, G_loss_curr = sess.run([G_solver, G_loss],
                                          feed_dict={CG: Y_mb,
                                                     Z: sample_Z(batch_size, seq_length, latent_dim,
                                                                 use_time=use_time)})
            return G_loss_curr

        def train_discriminator(batch_idx, offset):
            # update the discriminator
            for d in range(D_rounds):
                # using same input sequence for both the synthetic data and the real one,
                # probably it is not a good idea...
                X_mb, Y_mb = get_batch(train_seqs, train_targets, batch_size, batch_idx + d + offset)
                _, D_loss_curr = sess.run([D_solver, D_loss],
                                          feed_dict={CD: Y_mb, CG: Y_mb, X: X_mb,
                                                     Z: sample_Z(batch_size, seq_length, latent_dim,
                                                                 use_time=use_time)})

            return D_loss_curr

        for num_epoch in range(num_epochs):
            start_epoch = time.time()
            # we use D_rounds + G_rounds batches in each iteration
            for batch_idx in range(0, int(len(train_seqs) / batch_size) - (D_rounds + G_rounds), D_rounds + G_rounds):
                # we should shuffle the data instead
                if num_epoch % 2 == 0:
                    G_loss_curr = train_generator(batch_idx, 0)
                    D_loss_curr = train_discriminator(batch_idx, G_rounds)
                else:
                    D_loss_curr = train_discriminator(batch_idx, 0)
                    G_loss_curr = train_generator(batch_idx, D_rounds)

            t = time.time() - t0
            print(num_epoch, '\t', D_loss_curr, '\t', G_loss_curr, '\t', t)

            # record/visualise
            trace.write(str(num_epoch) + ' ' + str(D_loss_curr) + ' ' + str(G_loss_curr) + ' ' + str(t) + '\n')
            if num_epoch % 10 == 0:
                trace.flush()

            vis_sample = sess.run(G_sample, feed_dict={Z: vis_z, CG: Y_mb_vis})

            plotting.vis_FEM_downsampled(vis_sample, seq_length, folder_to_save_to=r"C:\Users\uvuik\Desktop\Test",
                                         identifier=identifier, idx=num_epoch+1)

            # save synthetic data
            if num_epoch % 10 == 0:
                # generate synthetic dataset
                gen_samples = []
                labels_gen_samples = []
                print(int(len(train_seqs) / batch_size))
                for batch_idx in range(int(len(train_seqs) / batch_size)):
                    X_mb, Y_mb = get_batch(train_seqs, train_targets, batch_size, batch_idx)
                    z_ = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
                    gen_samples_mb = sess.run(G_sample, feed_dict={Z: z_, CG: Y_mb})
                    gen_samples.append(gen_samples_mb)
                    labels_gen_samples.append(Y_mb)
                    print(batch_idx)

                for batch_idx in range(int(len(vali_seqs) / batch_size)):
                    X_mb, Y_mb = get_batch(vali_seqs, vali_targets, batch_size, batch_idx)
                    z_ = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
                    gen_samples_mb = sess.run(G_sample, feed_dict={Z: z_, CG: Y_mb})
                    gen_samples.append(gen_samples_mb)
                    labels_gen_samples.append(Y_mb)

                gen_samples = np.vstack(gen_samples)
                labels_gen_samples = np.vstack(labels_gen_samples)
                wd = ''
                with open(output_folder + '/samples_' + identifier + '_' + str(num_epoch) + '.pk', 'wb') as f:
                    pickle.dump(file=f, obj=gen_samples)

                with open(output_folder + '/labels_' + identifier + '_' + str(num_epoch) + '.pk', 'wb') as f:
                    pickle.dump(file=f, obj=labels_gen_samples)

                # save the model used to generate this dataset
                model.dump_parameters(identifier + '_' + str(num_epoch), sess)
            end_time = time.time()
            print(f'Epoche {num_epoch} hat {end_time-start_epoch}s benötigt.')


if __name__ == '__main__':
    folderpath = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\Augmented\Roorda_Augmented"
    run_training(folderpath, 'Roorda', 3)
    load_dataset(r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley", dataset_name='Roorda', seq_length=3)