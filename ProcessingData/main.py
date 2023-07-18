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
from ProcessingData.StatisticalEvaluation.Evaluation import Evaluation
from ProcessingData.StatisticalEvaluation.Microsaccades import Microsaccades
from Visualize import Visualize as Vis
from Preprocessing.Filtering import Filtering as Filt
from StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection


def get_constants(dataset_name):
    if dataset_name == "Roorda":
        return {"Name":"Roorda", "f": 1920, "x_col": 'xx', "y_col": 'yy', "time_col": 'TimeAxis', "ValScaling": 1/60, "TimeScaling": 1,
                'BlinkID': 3, 'Annotations': 'Flags', 'file_pattern': "\d{5}[A-Za-z]_\d{3}\.csv", 'rm_blink':False}
    elif dataset_name == "GazeBase":
        return {"Name": "GazeBase", "f": 1000, "x_col": 'x', "y_col": 'y', "time_col": 'n', "ValScaling": 60,
                "TimeScaling": 1 / 1000, 'BlinkID': -1, 'Annotations': 'lab', 'rm_blink':False} # Einheiten für y-Kooridnate ist in dva (degrees of vision angle)

def read_allgeier_data(folder_path, filenr=0):
    files = list(glob.glob(folder_path + "/*.txt"))
    #Create Dataframe
    df = pd.read_csv(files[filenr], names=['t','x_µm','y_µm'], header=None)
    const_dict = {"Name":"Allgeier", "f": 1/((df['t'].iloc[-1]-df['t'].iloc[0])/len(df)), "x_µm": 'x_µm', "y_µm": 'y_µm', "time_col": 't', "ValScaling": 1, "TimeScaling": 1,
                'BlinkID': None, 'Annotations': None, 'file_pattern': "/.+\.txt", 'rm_blink' : False}
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
        data, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        if resample:
            data, const_dict = Interpolation.resample(data, const_dict, rs_freq)
        micsac_detec2 = Microsaccades.find_micsac(data, const_dict, mindur=mindur, vfac=vfac)
        if const_dict['Name'] == "Roorda":
            data = Interpolation.convert_arcmin_to_dva(data, const_dict)
        micsac_detec = Microsaccades.find_micsac(data, const_dict, mindur=mindur, vfac=vfac)
        if len(micsac_detec[0]) != len(micsac_detec2[0]):
            print(len(micsac_detec[0])-len(micsac_detec2[0]), file)
        blink_rm, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        micsac_annot = Microsaccades.get_roorda_micsac(blink_rm)
        print(f'Es wurden {len(micsac_detec[0])} detektiert.\nEigentlich sind {len(micsac_annot)} vorhanden.')
        detected_micsac[Path(file).stem] = len(micsac_annot)#(micsac_detec[0])
    if save:
        print('Evaluation done')
        data_name = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\MicSacDetected.xlsx"
        save_dict_to_excel(detected_micsac, data_name)
    return data, micsac_detec[0]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    roorda_folder = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"

    roorda_test_file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\20109R_003.csv"

    roorda_data = pd.read_csv(roorda_test_file)
    const_roorda = get_constants("Roorda")
    roorda_files = get_files_with_pattern(roorda_folder, const_roorda['file_pattern'])
    allgeier_folder = r"\\os.lsdf.kit.edu\iai-projects\iai-aida\Daten_Allgeier\fixational_eye_movement_trajectories"
    gb_file = r"C:\Users\fanzl\bwSyncShare\Documents\GazeBase_v2_0\Fixation_Only\S_1006_S1_FXS.csv"
    gb_folder = ''
    const_gb = get_constants('GazeBase')
    gb_data = pd.read_csv(gb_file)
    gb_data, const_gb = Interpolation.dva_to_arcmin(gb_data, const_gb)
    #a, const_a = read_allgeier_data(allgeier_folder, 29)
    #plot_ds_comparison(roorda_data, const_roorda, a, const_a)
    blink_rm, const_dict = Interpolation.remove_blink_annot(roorda_data, const_roorda)
    #micsac_detec = Microsaccades.find_micsac(a, const_roorda)
    #intermicsac, dur, amplitudes, velocity = Evaluation.get_micsac_statistics(a, const_roorda, micsac_detec)
    #Evaluation.plot_prob_dist(intermicsac, 'Intermikrosakkadischer Abstand')
    #Evaluation.plot_prob_dist(dur, 'Dauer der jeweiligen Mikrosakkaden')
    #Evaluation.plot_prob_dist(amplitudes, 'Amplituden der Mikrosakkaden in arcmin')
    #Evaluation.plot_prob_dist(velocity, 'Geschwindigkeiten in arcmin/s')
    #b = Evaluation.intermicrosacc_len(a, const_roorda, micsac_detec)
    #micsac_annot = Microsaccades.get_roorda_micsac(a)
    #b2 = Evaluation.intermicrosacc_len(const_roorda, micsac_annot)
    #Plot both
    #Vis.plot_microsacc(a, const_roorda, title='Eye Trace in x- and y-Position', micsac=micsac_detec,
    #                   micsac2=micsac_annot,
    #                   color=['orange', 'blue', 'green', 'red', 'grey', 'black'], thickness=2,
    #                   legend=['x', 'y', 'Onset detect', 'Offset detect', 'Onset annot', 'Offset annot'])

    Vis.plot_xy(blink_rm[0:round(6000/1000*const_roorda['f'])],const_roorda,color=['b','orange'], labels=['x-Achse', 'y-Achse'], title='Menschliche Augenbewegung während der Fixation')
    drift_only_wo_micsac = EventDetection.drift_only_wo_micsac(blink_rm, const_dict)
    drift_only = EventDetection.drift_only(blink_rm, const_dict)
    drift_interpolated = EventDetection.drift_interpolated(blink_rm, const_dict)
    tremor_only = EventDetection.tremor_only(blink_rm, const_dict)
    filtered_drift = EventDetection.drift_only(blink_rm, const_dict)
    micsac_only = EventDetection.micsac_only(blink_rm, const_dict)
    #Evaluate all
    all_micsac_amp = []
    all_intermicsac = []
    all_micsac_dur = []
    all_micsac_vel = []
    all_drift_amp = []
    all_drift_vel = []
    all_tremor_amp = []
    all_tremor_vel = []
    for file in roorda_files:
        data = pd.read_csv(file)
        blink_rm, const_dict = Interpolation.remove_blink_annot(data, const_roorda)
        micsac_detec = Microsaccades.find_micsac(blink_rm, const_dict)
        intermicsac, dur, amplitudes, velocity = Evaluation.get_micsac_statistics(blink_rm, const_dict, micsac_detec)
        amp_drift, vel_drift = Evaluation.get_drift_statistics(blink_rm, const_dict)
        amp_tremor, vel_tremor = Evaluation.get_tremor_statistics(blink_rm, const_dict)
        all_micsac_amp.extend(amplitudes)
        all_intermicsac.extend(intermicsac)
        all_micsac_dur.extend(dur)
        all_micsac_vel.extend(velocity)
        all_drift_amp.extend(amp_drift)
        all_drift_vel.extend(vel_drift)
        all_tremor_amp.extend(amp_tremor)
        all_tremor_vel.extend(vel_tremor)
    print('Done')
    all_statistics = (all_micsac_amp, all_intermicsac, all_micsac_dur, all_micsac_vel, all_drift_amp, all_drift_vel, all_tremor_amp, all_tremor_vel)
    max_length = max(len(lst) for lst in all_statistics)

    #with open(r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\Statistics(mindur=10,vfac=21).csv", 'w',
    #     newline='') as csvfile:
    '''
        writer = csv.writer(csvfile)
        writer.writerow(
            ['all_micsac_amp', 'all_intermicsac', 'all_micsac_dur', 'all_micsac_vel', 'all_drift_amp', 'all_drift_vel',
             'all_tremor_amp', 'all_tremor_vel'])
        for i in range(max_length):
            try:
                row = [float(lst[i]) if i < len(lst) else '' for lst in all_statistics]
            except TypeError:
                row = ['' for _ in all_statistics]
            writer.writerow(row)
        '''

    Vis.plot_prob_dist([i*60 for i in all_tremor_amp], 'Amplitude of tremor', 'Amplitude in arcsec')#Why does this lead to an error?
    Vis.plot_prob_dist(all_micsac_amp, 'Amplitudes of Microsaccades', 'Amplitude in arcmin')
    Vis.plot_prob_dist(all_intermicsac, 'Intermicrosaccadic Duration', 'Duration in s')
    Vis.plot_prob_dist(all_micsac_vel, 'Velocity of Microsaccades', 'Velocity in °/s')
    Vis.plot_prob_dist(all_drift_amp, 'Amplitude of drift', 'Amplitude in arcmin')

    amp_drift, vel_drift = Evaluation.get_drift_statistics(blink_rm, const_dict)
    amp_tremor, vel_tremor = Evaluation.get_tremor_statistics(blink_rm, const_dict)
    fft, fftfreq = Filt.fft_transform(filtered_drift, const_roorda, 'x_col')
    Vis.plot_fft(fft, fftfreq)
    Vis.plot_xy(filtered_drift, const_roorda)







    gb_data = pd.read_csv(gb_file)

    Vis.plot_xy_trace(roorda_data[0:8000], const_roorda, label='Eye trace', color='orange')
    #data = gb_data
    #const_dict = const_gb

    #Working with Roorda_data
    roorda_folder = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    df, micsac = detect_all_micsac(roorda_folder, mindur=10, vfac=21, const_dict=const_roorda,resample=False, save=True)
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
