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
import plotting

import model

def test_x_coordinate(folderpath):


def load_dataset(folderpath):
    csv_files = [f for f in os.listdir(folderpath) if f.endswith('.csv')]
    num_samples = len(csv_files)
    samples_x = []
    samples_y = []
    for i, file in enumerate(csv_files):
        filepath = os.path.join(folderpath, file)
        df = pd.read_csv(filepath)
        if i == 0:
            col_x = len(df['xx'])
            col_y = len(df['yy'])
        samples_x.append(df['xx'].values)
        samples_y.append(df['yy'].values)
    #Convert List to ndarray
    samples_x = np.array(samples_x)
    samples_y = np.array(samples_y)

    #reshape
    samples_x = samples_x.reshape((num_samples, col_x, 1))
    samples_y = samples_y.reshape((num_samples, col_y, 1))
    return samples_x, samples_y

def folder_input():
    # Ask the user for the folder path
    folder_path = input("Please enter the folder path: ")

    # Check if the entered path exists
    while not os.path.exists(folder_path):
        print("The entered folder path does not exist. Please check it again.")
        folder_path = input("Please enter the correct folder path: ")

    return folder_path

def run_training():
    # run experiment 5 times
    identifiers = ['FEM_synthetic_dataset' + str(i) for i  in range(2,5)]
    for identifier in identifiers:
        tf.compat.v1.reset_default_graph()
        print(f'loading data for {identifier}...')

        samples, labels = data_utils.FEM_task()
        new_shape = (-1,16,4)#TODO: Shape anpassen, was auch immer hier gemacht wird
        train_seqs = samples['train'].reshape(new_shape)


