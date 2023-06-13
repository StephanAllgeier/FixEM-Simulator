import csv
import glob
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from Preprocessing.Interpolation import Interpolation
from ProcessingData.StatisticalEvaluation.Microsaccades import Microsaccades
from Visualize import Visualize as Vis
from Preprocessing.Filtering import Filtering as Filt
from StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection


def get_constants(dataset_name):
    if dataset_name == "Roorda":
        return {"Name":"Roorda", "f": 1920, "x_col": 'xx', "y_col": 'yy', "time_col": 'TimeAxis', "ValScaling": 1, "TimeScaling": 1,
                'BlinkID': 3, 'Annotations': 'Flags', 'file_pattern': "\d{5}[A-Za-z]_\d{3}\.csv"}
    elif dataset_name == "GazeBase":
        return {"Name": "GazeBase", "f": 1000, "x_col": 'x', "y_col": 'y', "time_col": 'n', "ValScaling": 60,
                "TimeScaling": 1 / 1000, 'BlinkID': -1, 'Annotations': 'lab'} # Einheiten für y-Kooridnate ist in dva (degrees of vision angle)

def read_allgeier_data(folder_path):
    files = list(glob.glob(folder_path + "/*.txt"))
    #Create Dataframe
    df = pd.read_csv(files[0], names=['t','x_µm','y_µm'], header=None)
    print(df)
    const_dict = {"Name":"Allgeier", "f": 1/((df['t'].iloc[-1]-df['t'].iloc[0])/len(df)), "x_µm": 'x_µm', "y_µm": 'y_µm', "time_col": 't', "ValScaling": 1, "TimeScaling": 1,
                'BlinkID': None, 'Annotations': None, 'file_pattern': "/.+\.txt"}
    return df, const_dict

def plot_ds_comparison(df1, const1, df2, const2):
    name1 = const1['Name']
    name2 = const2['Name']
    if name1 != 'Allgeier':
        df1, const1 = Interpolation.arcmin_to_µm(df1, const1)
    if name2 != 'Allgeier':
        df2, const2 = Interpolation.arcmin_to_µm(df1, const1)
    Vis.plot_xy_µm(df1, const1, color=['black', 'green'], labels=[f'x_{name1}', f'y_{name1}'])
    Vis.plot_xy_µm(df2, const2, color=['orange', 'red'], labels=[f'x_{name2}', f'y_{name2}'])
    print('Done')
def get_files_with_pattern(folder_path, pattern):
    file_list = []
    regex_pattern = re.compile(pattern)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and regex_pattern.match(filename):
            file_list.append(file_path)
    return file_list

def save_dict_to_csv(data_dict, file_path):
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File', 'MicSacCount'])  # Header schreiben
        for key, value in data_dict.items():
            writer.writerow([key, value])
def save_dict_to_excel(data_dict, file_path):
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Value'])
    df.index.name = 'Key'
    df.to_excel(file_path)

def detect_all_micsac(folderpath, const_dict,mindur, vfac, resample=False, rs_freq=1000, save=False):
    files = get_files_with_pattern(Path(folderpath), pattern=const_dict['file_pattern'])
    detected_micsac = {}
    for file in files:
        df = pd.read_csv(Path(file))
        data = Interpolation.remove_blink_annot(df, const_dict)
        if resample:
            data, const_dict = Interpolation.resample(data, const_dict, rs_freq)
        micsac_detec2 = Microsaccades.find_micsac(data, const_dict, mindur=mindur, vfac=vfac)
        if const_dict['Name'] == "Roorda":
            data = Interpolation.convert_arcmin_to_dva(data, const_dict)
        micsac_detec = Microsaccades.find_micsac(data, const_dict, mindur=mindur, vfac=vfac)
        if len(micsac_detec[0]) != len(micsac_detec2[0]):
            print(len(micsac_detec[0])-len(micsac_detec2[0]), file)
        micsac_annot = Microsaccades.get_roorda_micsac(Interpolation.remove_blink_annot(df, const_dict))
        print(f'Es wurden {len(micsac_detec[0])} detektiert.\nEigentlich sind {len(micsac_annot)} vorhanden.')
        detected_micsac[Path(file).stem] = len(micsac_annot)#(micsac_detec[0])
    if save:
        print('Evaluation done')
        data_name = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\MicSacDetected.xlsx"
        save_dict_to_excel(detected_micsac, data_name)
    return data, micsac_detec[0]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # roorda_files = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    roorda_test_file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\20109R_003.csv"
    roorda_data = pd.read_csv(roorda_test_file)
    const_roorda = get_constants("Roorda")
    allgeier_folder = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\Allgeier, Stephan"
    #a, const_a = read_allgeier_data(allgeier_folder)
    filtered,b,c = EventDetection.drift_only(roorda_data, const_roorda)
    fft, fftfreq = Filt.fft_transform(filtered, const_roorda, 'x_col')
    Vis.plot_fft(fft, fftfreq)
    #plot_ds_comparison(roorda_data, const_roorda, a, const_a)
    #gb_data = pd.read_csv(gb_file)
    #const_gb = get_constants("GazeBase")
    #
    #data = gb_data
    #const_dict = const_gb

    #Working with Roorda_data
    roorda_folder = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    df, micsac = detect_all_micsac(roorda_folder, mindur=25, vfac=21, const_dict=const_roorda,resample=False, save=True)
    # Visualize both:
    #Vis.plot_xy(data, const_dict)
    #Vis.plot_xy(data, const_dict, colors=['red', 'orange'], labels=['x Roorda', 'y Roorda'])
    #Vis.plot_xy(gab_data, const_gb, colors=['blue', 'violet'], labels=['x GazeBase', 'y GazeBase'])

    #Microsaccades according to Roorda:    roorda_micsac = Microsaccades.get_roorda_micsac(roorda_data)
    #Vis.print_microsacc(roorda_data, const_roorda, roorda_micsac)
    '''
    # Filtering Bandpass
    Drift_c = EventDetection.filter_drift(cubic, const_dict)
    Tremor_c = EventDetection.filter_tremor(cubic, const_dict)
    Drift_p = EventDetection.filter_drift(piece_poly, const_dict)
    Tremor_p = EventDetection.filter_tremor(piece_poly, const_dict)
    Drift = EventDetection.filter_drift(spliced, const_dict)
    Tremor = EventDetection.filter_tremor(spliced, const_dict)

    #Plotting Tremor/Drift and belonging Frequencie Spectrum

    fft2, fftfreq2 = Filt.fft_transform(Drift, const_dict, 'x_col')
    Vis.plot_fft(fft2, fftfreq2)
    micsacc = EventDetection.find_micsacc(spliced, const_dict, mindur=12)

    Vis.print_microsacc(spliced, const_dict, micsacc)
    '''
