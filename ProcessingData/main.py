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
        return {"f": 1920, "x_col": 'xx', "y_col": 'yy', "time_col": 'TimeAxis', "ValScaling": 1, "TimeScaling": 1,
                'BlinkID': 3, 'Annotations': 'Flags'}
    elif dataset_name == "GazeBase":
        return {"f": 1000, "x_col": 'x', "y_col": 'y', "time_col": 'n', "ValScaling": 60,
                "TimeScaling": 1 / 1000, 'BlinkID': -1, 'Annotations': 'lab'}  # Einheiten für y-Kooridnate ist in dva (degrees of vision angle)
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

def get_events(df, const_dict, msac_mindur=4):
    return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # roorda_files = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    roorda_test_file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\20109R_003.csv"
    roorda_data = pd.read_csv(roorda_test_file)
    const_roorda = get_constants("Roorda")

    gb_file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\Testing\S_1001_S1_FXS.csv"
    gb_data = pd.read_csv(gb_file)
    const_gb = get_constants("GazeBase")
    #
    data = gb_data
    const_dict = const_gb

    #Working with Roorda_data
    roorda_folder = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    roorda_filepattern = "\d{5}[A-Za-z]_\d{3}\.csv"
    roorda_files = get_files_with_pattern(Path(roorda_folder), pattern=roorda_filepattern)
    roorda_count_dict = {}
    for file in roorda_files:
        micsac_count = Microsaccades.count_micsac_annot(pd.read_csv(Path(file)))
        roorda_count_dict[Path(file).stem] = micsac_count
    #data_name=r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\MicSacEval.csv"
    #save_dict_to_csv(roorda_count_dict, data_name)

    detected_micsac = {}
    for file in roorda_files:
        interpolated = Interpolation.remove_blink_annot(pd.read_csv(Path(file)), const_roorda)
        interpol = Interpolation.convert_arcmin_to_dva(interpolated, const_roorda)
        filtered = EventDetection.filter_drift(interpol, constant_dict=const_roorda, highcut=40, order=5)
        micsac= Microsaccades.find_micsacc(interpol, const_roorda, mindur=10)
        detected_micsac[Path(file).stem] = len(micsac[0])
    print('Evaluation done')
    data_name = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\MicSacDetected.xlsx"
    save_dict_to_excel(detected_micsac, data_name)

    # Visualize both:
    #Vis.plot_xy(data, const_dict)
    #Vis.plot_xy(data, const_dict, colors=['red', 'orange'], labels=['x Roorda', 'y Roorda'])
    #Vis.plot_xy(gab_data, const_gb, colors=['blue', 'violet'], labels=['x GazeBase', 'y GazeBase'])

    #Microsaccades according to Roorda:
    roorda_micsac = Microsaccades.get_roorda_micsac(roorda_data)
    Vis.print_microsacc(roorda_data, const_roorda, roorda_micsac)

    # Interpolation
    cubic = Interpolation.interp_cubic(data, const_dict)
    piece_poly = Interpolation.interp_monocub(data, const_dict)
    spliced = Interpolation.splice_together(data, const_dict)
    blink_removed = Interpolation.remove_blink(data, const_dict, 10, remove_start=True, remove_end = False,remove_start_time= 20, remove_end_time= 0)
    # FastFourierTransformation
    fft_spliced, fftfreq_spliced = Filt.fft_transform(spliced, const_dict, 'x_col')
    Vis.plot_fft(fft_spliced, fftfreq_spliced)

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
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
