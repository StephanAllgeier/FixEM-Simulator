import csv
import glob
import json
import os
import re
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from ProcessingData.Preprocessing.Interpolation import Interpolation
from ProcessingData.StatisticalEvaluation import Evaluation
from ProcessingData.Preprocessing.Augmentation import Augmentation
from ProcessingData.StatisticalEvaluation.Microsaccades import Microsaccades
from ProcessingData.Visualize import Visualize as Vis
from ProcessingData.Preprocessing.Filtering import Filtering as Filt, Filtering
from ProcessingData.StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection


# Function to return dataset constants as dict
def get_constants(dataset_name):
    if dataset_name == "Roorda":
        return {"Name": "Roorda", "f": 1920, "x_col": 'xx', "y_col": 'yy', "time_col": 'TimeAxis', "ValScaling": 1 / 60,
                "TimeScaling": 1,
                'BlinkID': 3, 'Annotations': 'Flags', 'file_pattern': "\d{5}[A-Za-z]_\d{3}\.csv", 'rm_blink': False}
    elif dataset_name == "GazeBase":
        return {"Name": "GazeBase", "f": 1000, "x_col": 'x', "y_col": 'y', "time_col": 'n', "ValScaling":1 ,
                "TimeScaling": 1 / 1000, 'BlinkID': -1, 'Annotations': 'flags',
                'rm_blink': False, 'SakkID':2, 'file_pattern':"[A-Za-z0-9]+\.csv"}  # Einheiten für y-Kooridnate ist in dva (degrees of vision angle)
    elif dataset_name == "OwnData":
        return {"Name": "OwnData", "f": 500, "x_col": 'x', "y_col": 'y', "time_col": 'Time', "ValScaling": 1,
                "TimeScaling": 1, 'BlinkID': None, 'Annotations': 'lab', 'rm_blink': False}

# Function to get all files with given pattern as regex
def get_files_with_pattern(folder_path, pattern):
    file_list = []
    regex_pattern = re.compile(pattern)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and regex_pattern.match(filename):
            file_list.append(file_path)
    return file_list

# Return all CSV-files from folderpath
def get_csv_files_in_folder(folderpath):
    csv_files = []
    for filename in os.listdir(folderpath):
        if filename.endswith(".csv"):
            file_path = os.path.join(folderpath, filename)
            csv_files.append(file_path)
    return csv_files


def save_dict_to_csv(data_dict, file_path):
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File', 'MicSacCount'])  # Write Header
        for key, value in data_dict.items():
            writer.writerow([key, value])


def save_dict_to_excel(data_dict, file_path):
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Value'])
    df.index.name = 'Key'
    df.to_excel(file_path)

# Function to label micsacs in given files from folderpath
def label_micsac(folderpath,const_dict, mindur, vfac):
    files_pattern = rf"{folderpath}\*.csv"
    files = glob.glob(files_pattern)
    for file in files:
        df = pd.read_csv(Path(file))
        df['flags'] = 0
        micsac = Microsaccades.find_micsac(df, const_dict, mindur=mindur, vfac=vfac)
        micsac_list = micsac[0]
        micsac_onsets = [microsac[0] for microsac in micsac_list]
        micsac_offsets = [microsac[1] for microsac in micsac_list]
        assert len(micsac_onsets) == len(micsac_offsets), 'Wrong dimensions'
        for onset, offset in zip(micsac_onsets, micsac_offsets):
            df.loc[onset:offset, 'flags'] = 1
        #speichern
        df_filtered = df.iloc[:, 1:]
        df_filtered.to_csv(file)
        print(f"{len(micsac[0])} mikrosaccades have been found within the timewindow of {len(df)/const_dict['f']}s")

# Function to detect microsaccades
def detect_all_micsac(folderpath, const_dict, mindur, vfac, save=False):
    files = get_files_with_pattern(Path(folderpath), pattern=const_dict['file_pattern'])
    detected_micsac = {}
    for file in files:
        df = pd.read_csv(Path(file))
        data, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        micsac_detec2 = Microsaccades.find_micsac(data, const_dict, mindur=mindur, vfac=vfac)
        if const_dict['Name'] == "Roorda":
            data = Interpolation.convert_arcmin_to_dva(data, const_dict)
        micsac_detec = Microsaccades.find_micsac(data, const_dict, mindur=mindur, vfac=vfac)
        if len(micsac_detec[0]) != len(micsac_detec2[0]):
            print(len(micsac_detec[0]) - len(micsac_detec2[0]), file)
        blink_rm, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        micsac_annot = Microsaccades.get_roorda_micsac(blink_rm)
        print(f'Es wurden {len(micsac_detec[0])} detektiert.\nEigentlich sind {len(micsac_annot)} vorhanden.')
        detected_micsac[Path(file).stem] = len(micsac_detec[0])  # (micsac_detec[0])
    if save:
        print('Evaluation done')
        data_name = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\MicSacDetected.xlsx"
        save_dict_to_excel(detected_micsac, data_name)
    return data, micsac_detec[0]

# Loading json-files
def get_json_file(folderpath):
    json_files = []
    for filename in os.listdir(folderpath):
        if filename.endswith('.json'):
            json_files.append(os.path.join(folderpath, filename))
    return json_files

# Funciton to evaluate intermic duration as histogram with linear logfit plot
def evaluate_intermic_hist(json_path, rangelim = 2.5):
    Evaluation.Evaluation.generate_histogram_with_logfit(json_path, range = (0,rangelim))

# Function to remove blink from GazeBase. Additionally remove first second of recording
def remove_blink_gb(data_inp, const_gb, timecutoff):
    data = data_inp.iloc[1000:] #remove first second of recording
    data.reset_index(drop=True, inplace=True)
    indices_to_remove = data[(data[const_gb['Annotations']] == -1) | (data[const_gb['Annotations']] == 2)].index

    # Collect indixes to be removed. Adding additional timeout of sequences
    indices_to_remove_extended = []
    for idx in indices_to_remove:
        start_idx = idx - timecutoff
        end_idx = idx + timecutoff + 1
        if start_idx<0:
            start_idx = 0
        if end_idx > len(data):
            end_idx = len(data)-1
        indices_to_remove_extended.extend(range(start_idx,end_idx))
        indices_to_remove_extended = list(set(indices_to_remove_extended))
    data_cleaned = data.drop(indices_to_remove_extended)
    # Set indices
    data_cleaned.reset_index(drop=True, inplace=True)
    data_cleaned['n'] = data_cleaned.index
    return data_cleaned

# Function to augment datasets. This function combines removal of blink, creating a new timeseries of flipped values and
# one with gaussian noise added.
def augmentation(datafolder, savefolder, const_dict):
    files = get_csv_files_in_folder(datafolder)
    for file in files:
        data = pd.read_csv(file)
        if const_dict['Name'] == "GazeBase":
            removed_blink = remove_blink_gb(data, const_dict, 15)
        elif const_dict['Name'] == "Roorda":
            removed_blink, const_gb = Interpolation.remove_blink_annot(data, const_dict)
        new_cols = {const_dict['x_col']:'x', const_dict['y_col']:'y', const_dict['Annotations']:'flags'}
        flipped_ts = Augmentation.flip_dataframe(removed_blink, const_dict).rename(columns=new_cols)
        removed_blink = removed_blink.rename(columns=new_cols)
        removed_blink.to_csv(fr"{savefolder}\{Path(file).stem}.csv")
        flipped_ts.to_csv(fr"{savefolder}\{Path(file).stem}_flipped.csv")
        gaussian_frame = Augmentation.add_gaussian_noise(removed_blink, const_dict).rename(columns=new_cols)
        gaussian_frame.to_csv(fr"{savefolder}\{Path(file).stem}_gaussian.csv")
    return None


# Function to loop through all json-files of folderpath_to_jsons. Extract top 30% of lowest HD
def get_best_HD(folderpath_to_jsons, feature, compare_json_file, normalize_01, output_folder=None, filename = "ParameterInput_IntermicDur_SimDur=30s"):
    simulation_dict={}
    json_files = get_json_file(folderpath_to_jsons)
    with open(compare_json_file,'r') as compare:
        compare_data = json.load(compare)[feature]
    for file in json_files:
        with open(file, 'r') as open_file:
            data = json.load(open_file)[feature]
        file_name = Path(file).stem.replace('_n=50','').split(',')
        sim_rate = file_name[0].split('=')[1]
        cells_per_deg=file_name[1].split('=')[1]
        relaxation_rate = file_name[2].split('=')[1]
        hc=file_name[3].split('=')[1]
        HD = Evaluation.Evaluation.normalized_histogram_difference(data, compare_data, num_bins= 50, normalize_01=normalize_01)

        name = f'simulation rate={sim_rate}, cells per degree={cells_per_deg}, relaxation rate={relaxation_rate}, h_crit={hc}'
        simulation_dict.update({name: HD})
    # Sammle alle HD-Werte aus dem Dictionary
    hd_vals = list(simulation_dict.values())
    # Sort HD-values
    hd_vals.sort()
    threshold_index = int(0.3 * len(hd_vals))
    threshold= hd_vals[threshold_index]
    # Durchsuche das Dictionary nach Namen mit HD-Werten kleiner oder gleich dem Schwellenwert
    valid_hd = [(name, hd) for name, hd in simulation_dict.items() if hd <= threshold]
    columns= ['simulation rate','cells per degree','relaxation rate','h_crit', f'Mean_{feature}', f'Median_{feature}', f'sigma_{feature}', 'HD']
    my_list=[]
    for element in valid_hd:

        split_element = element[0].split(',')
        simrate = split_element[0].split('=')[1]
        cells =split_element[1].split('=')[1]
        relax =split_element[2].split('=')[1]
        hc = split_element[3].split('=')[1]
        file_name = rf'{folderpath_to_jsons}\AmpDurNum_simrate={simrate},Cells={cells},relaxationrate={relax},hc={hc}_n=50.json'
        hd = round(element[1], 3)
        with open(file_name, 'r') as open_file2:
            data2 = json.load(open_file2)[feature]
        mean = np.mean(data2)
        median = np.median(data2)
        std_dev = np.std(data2)
        my_list.append([simrate,cells,relax,hc,mean,median,std_dev,hd])
        file_name_new = f'HD={hd}_simulation_rate={simrate}_CellsPerDegree={cells}_RelaxationRate={relax}_HCrit={hc}.json'
        new_file_path = f"{Path(folderpath_to_jsons)}/{output_folder}/{file_name_new}"


        shutil.copyfile(Path(file_name), new_file_path)
    df = pd.DataFrame(my_list, columns=columns)
    excel_datei = rf"{folderpath_to_jsons}\{filename}.xlsx"
    df.to_excel(excel_datei, index=False)
    csv_datei = rf"{folderpath_to_jsons}\{filename}.csv"
    df.to_csv(csv_datei, index=False)


# This function creates a dual histogram plot with log scale, comparing a common dataset (compare_filepath)
# with two others (file1 and file2) based on a specified feature, and saves the plot as a JPEG file.
def create_histogram_dual_w_HD(compare_filepath, file1, file2, feature, xlabel, compare_label, label1, label2, savefigpath):
    with open(
            compare_filepath,
            'r') as comp:
        compare_data = json.load(comp)[feature]
    with open(file1, 'r') as comp2:
        varlist2_1 = json.load(comp2)[feature]
    with open(file2, 'r') as comp3:
        varlist2_2 = json.load(comp3)[feature]
    savefig = f"{Path(savefigpath).parent}/DualHistogramLog_intermicsac.jpeg"
    Evaluation.hist_subplot_w_histdiff_log(compare_data, varlist2_1, compare_label, label1, savefig, xlabel,
    #                                       range_limits=(0, 2.5), normalize_01=False, title=title)
    Evaluation.dual_hist_subplot_w_histdiff_log(compare_data, varlist2_1, varlist2_2, compare_label, label1, label2, savefig, xlabel, range_limits=(0, 2.5), normalize_01=False)

# This function creates individual histograms for datasets in a specified folder (folderpath),
# comparing each dataset with a common dataset (compare_filepath). The histograms are based on a specified feature and
# are saved as JPEG files.
def create_histogram_w_HD(compare_filepath, folderpath, feature, xlabel, dataset1name, dataset2name):
    with open(
            compare_filepath,
            'r') as comp:
        compare_data = json.load(comp)[feature]
    json_files = get_json_file(folderpath)
    for file in json_files:
        with open(file, 'r') as intermic_json:
            try:
                intermic_dur = json.load(intermic_json)[feature]
            except:
                pass
        hist_bins = 50
        savefigpath = file
        savefig = f"{savefigpath[:-5]}_histogram_intermicsac.jpeg"
        if not Path(savefig).is_file():
            Evaluation.Evaluation.dual_hist_w_histdiff(compare_data, intermic_dur,dataset1name, dataset2name, savefigpath, xlabel,range_limits=(0, 2.5), normalize_01=False)

# This function extracts microsaccade angles from CSV files in a specified folder, based on provided
# flags (ms_flag and drift_flag). It identifies microsaccades by flag sequences, calculates their amplitudes in the x
# and y directions, and computes corresponding angles. The function returns a list of angles for all identified
# microsaccades across the files. Any errors encountered during processing are printed, and the function continues
# with the next file.
def get_micsac_ang(folderpath, ms_flag, drift_flag):
    files = []
    for filename in os.listdir(folderpath):
        if filename.endswith(".csv"):
            file_path = os.path.join(folderpath, filename)
            files.append(file_path)
    angles = []
    for file_path in files:
        df = pd.read_csv(file_path)
        onset_indices = []
        amplitudes_x = []
        amplitudes_y = []
        try:
            for index,row in df.iterrows():
                if row['Flags'] == ms_flag and df.loc[index - 1, 'Flags'] == 2:
                    onset_indices.append(index)
                elif row['Flags'] == drift_flag and onset_indices:
                    onset_index = onset_indices.pop()
                    # Finde den Offset-Index als die letzte 1 in der Serie von 1en nach dem Onset
                    offset_index = df[df.index > onset_index]['Flags'].eq(2).idxmax() -1
                    microsaccade_df = df.loc[onset_index:offset_index]

                    max_x_index = microsaccade_df['xx'].idxmax()
                    min_x_index = microsaccade_df['xx'].idxmin()
                    max_y_index = microsaccade_df['yy'].idxmax()
                    min_y_index = microsaccade_df['yy'].idxmin()

                    # Überprüfe die Indizes von Maxima und Minima und passe das Vorzeichen an
                    amplitude_x = microsaccade_df.loc[max_x_index, 'xx'] - microsaccade_df.loc[min_x_index, 'xx']
                    amplitude_y = microsaccade_df.loc[max_y_index, 'yy'] - microsaccade_df.loc[min_y_index, 'yy']

                    # Überprüfe die Indizes von Maxima und Minima und passe das Vorzeichen an
                    if max_x_index < min_x_index:
                        amplitude_x *= -1
                    if max_y_index < min_y_index:
                        amplitude_y *= -1
                    amplitudes_x.append(amplitude_x)
                    amplitudes_y.append(amplitude_y)
                    angle = np.arctan2(amplitude_y, amplitude_x) * 180 / np.pi
                    angles.append(angle)
        except Exception as e:
            print(f"Error while working with file {file_path}: {e}")
            continue
    return angles

if __name__ == '__main__':
    ### Example Usages

    const_roorda = get_constants('Roorda')
    #folderpath_roorda = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\Roorda"
    #roorda_file = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\Roorda\10003L_001.csv"
    #const_gb = get_constants('GazeBase')
    #folderpath_gb = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\GazeBase"
    #gb_file =  r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\GazeBase\S_1001_S1_FXS.csv"

    # Augmentation
    #a =  augmentation(folderpath_gb, savefolder = r"C:\Users\uvuik\Desktop\Testfolder", const_dict = const_gb)

    #Creating histogram plots
    compare_filepath = r"C:\Users\uvuik\bwSyncShare\Abgabe IAI\Datasets\Roorda Vision Berkeley\MicsacFeatures.json"
    file1 = r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Evaluation20s\BestHD_IntermicDur\HD=3.969_simulation_rate=150_CellsPerDegree=10_RelaxationRate=0.085_HCrit=8.9.json"
    file2 = r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Evaluation20s\BestHD_IntermicDur\HD=4.103_simulation_rate=100_CellsPerDegree=10_RelaxationRate=0.1_HCrit=6.9.json"
    create_histogram_dual_w_HD(
        compare_filepath, file1, file2, 'IntermicDur', 'Intermikrosakkadische Intervalldauer in Sekunden [s]',
        label1='(f_sim=150Hz, L=21, epsilon=0.085, h_crit=8.9)', label2='(f_sim=100Hz, L=21, epsilon=0.1, h_crit=6.9)', compare_label='Roorda Lab', savefigpath=r"C:\Users\uvuik\Desktop\NewFolder")

    # Functions for plotting Histograms
    create_histogram_w_HD(
        compare_filepath,
        r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Evaluation30s\BestHD_IntermicDur",
        'IntermicDur', 'Intermikrosakkadischer Abstand in Sekunden [s]', 'Roorda Lab', 'Math. Modell')

    get_best_HD(r"C:\Users\uvuik\Desktop\TestOrdner40s", 'IntermicDur',
                compare_filepath, normalize_01=False, output_folder="BestHD_IntermicDur")

    # FFT and Cisualization of FFT
    data_gb = pd.read_csv(gb_file)
    fft, fftfreq = Filt.fft_transform(data_gb, const_gb, 'x_col', 'y_col')
    savepath_fft = r"C:\Users\uvuik\Desktop\FrequencyAnalysis\GazeBaseBlinkStitchedFFT.jpeg"
    Vis.plot_fft(fft, fftfreq, savepath_fft, title = "Fourier-Analyse einer FEM-Trajektorie\n des GazeBase-Datensatz")

    # Labeling micsacs in given dataset
    label_micsac(folderpath_gb,const_dict=const_gb, mindur=6, vfac=10)

    # Different preprocessing functions
    data = pd.read_csv(roorda_file, const_roorda)
    const_dict = const_roorda
    drift_only_wo_micsac = EventDetection.drift_only_wo_micsac(data, const_dict)
    drift_only = EventDetection.drift_only(data, const_dict)
    drift_interpolated = EventDetection.drift_interpolated(data, const_dict)
    tremor_only = EventDetection.tremor_only(data, const_dict)
    filtered_drift = EventDetection.drift_only(data, const_dict)
    micsac_only = EventDetection.micsac_only(data, const_dict)
