import csv
import glob
import json
import os
import re
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from ProcessingData.Preprocessing.Interpolation import Interpolation
from ProcessingData.StatisticalEvaluation import Evaluation
from ProcessingData.Preprocessing.Augmentation import Augmentation
from ProcessingData.StatisticalEvaluation.Microsaccades import Microsaccades
from ProcessingData.Visualize import Visualize as Vis
from ProcessingData.Preprocessing.Filtering import Filtering as Filt, Filtering
from ProcessingData.StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection


class DatasetConstants:
    def __init__(self, name, f, x_col, y_col, time_col, val_scaling, time_scaling, blink_id, annotations, rm_blink=False, sakk_id=None):
        self.Name = name
        self.f = f
        self.x_col = x_col
        self.y_col = y_col
        self.time_col = time_col
        self.ValScaling = val_scaling
        self.TimeScaling = time_scaling
        self.BlinkID = blink_id
        self.Annotations = annotations
        self.rm_blink = rm_blink
        self.SakkID = sakk_id

def get_constants(dataset_name):
    if dataset_name == "Roorda":
        return {"Name": "Roorda", "f": 1920, "x_col": 'xx', "y_col": 'yy', "time_col": 'TimeAxis', "ValScaling": 1 / 60,
                "TimeScaling": 1,
                'BlinkID': 3, 'Annotations': 'Flags', 'file_pattern': "\d{5}[A-Za-z]_\d{3}\.csv", 'rm_blink': False}
    elif dataset_name == "GazeBase":
        return {"Name": "GazeBase", "f": 1000, "x_col": 'x', "y_col": 'y', "time_col": 'n', "ValScaling":1 ,
                "TimeScaling": 1 / 1000, 'BlinkID': -1, 'Annotations': 'lab',
                'rm_blink': False, 'SakkID':2, 'file_pattern':"[A-Za-z0-9]+\.csv"}  # Einheiten für y-Kooridnate ist in dva (degrees of vision angle)
    elif dataset_name == "OwnData":
        return {"Name": "OwnData", "f": 500, "x_col": 'x', "y_col": 'y', "time_col": 'Time', "ValScaling": 1,
                "TimeScaling": 1, 'BlinkID': None, 'Annotations': 'lab', 'rm_blink': False}
def read_allgeier_data(folder_path, filenr=0):
    files = list(glob.glob(folder_path + "/*.txt"))
    # Create Dataframe
    df = pd.read_csv(files[filenr], names=['t', 'x_µm', 'y_µm'], header=None)
    const_dict = {"Name": "Allgeier", "f": 1 / ((df['t'].iloc[-1] - df['t'].iloc[0]) / len(df)), "x_µm": 'x_µm',
                  "y_µm": 'y_µm', "time_col": 't', "ValScaling": 1, "TimeScaling": 1,
                  'BlinkID': None, 'Annotations': None, 'file_pattern': "/.+\.txt", 'rm_blink': False}
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

def get_csv_files_in_folder(folder_path):
    csv_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            csv_files.append(file_path)
    return csv_files


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
        assert len(micsac_onsets) == len(micsac_offsets), 'UNGLEICH LANG'
        for onset, offset in zip(micsac_onsets, micsac_offsets):
            df.loc[onset:offset, 'flags'] = 1
        #speichern
        df_filtered = df.iloc[:, 1:]
        df_filtered.to_csv(file)


        print(f"Es wurden {len(micsac[0])} Mikrosakkaden gefunden in einem Zeitraum von {len(df)/const_dict['f']}s")

def detect_all_micsac(folderpath, const_dict, mindur, vfac, resample=False, rs_freq=1000, save=False):
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
            print(len(micsac_detec[0]) - len(micsac_detec2[0]), file)
        blink_rm, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        micsac_annot = Microsaccades.get_roorda_micsac(blink_rm)
        print(f'Es wurden {len(micsac_detec[0])} detektiert.\nEigentlich sind {len(micsac_annot)} vorhanden.')
        detected_micsac[Path(file).stem] = len(micsac_detec[0])  # (micsac_detec[0])
    if save:
        print('Evaluation done')
        data_name = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\MicSacDetected_new.xlsx"
        save_dict_to_excel(detected_micsac, data_name)
    return data, micsac_detec[0]

def get_json_file(folderpath):
    json_files = []
    for filename in os.listdir(folderpath):
        if filename.endswith('.json'):
            json_files.append(os.path.join(folderpath, filename))
    return json_files


def read_all_values_from_csv(folder_path, end, column_name):
    all_names = []
    csv_files = list(glob.glob(os.path.join(folder_path, f'*{end}')))

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        if column_name in df.columns:
            all_names.extend(df['Name'].tolist())

    return all_names
def evaluate_all_hist():
    all_json = get_json_file(
        r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\hc_3,4,10-30s, n=50")
    # for file in all_json:
    #    Evaluation.Evaluation.evaluate_json_hist(file)
def evaluate_intermic_hist(json_path, rangelim = 2.5):
    Evaluation.Evaluation.generate_histogram_with_logfit(json_path, range = (0,rangelim))

def remove_blink_gb(data_inp, const_gb, timecutoff):
    data = data_inp.iloc[1000:]
    data.reset_index(drop=True, inplace=True)
    indices_to_remove = data[(data[const_gb['Annotations']] == -1) | (data[const_gb['Annotations']] == 2)].index

    # Sammle die Indizes der Zeilen, die du entfernen möchtest (15 vorher und 15 danach)
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

    # Setze die Indizes neu
    data_cleaned.reset_index(drop=True, inplace=True)
    data_cleaned['n'] = data_cleaned.index
    return data_cleaned
def augmentation():
    gb_folder = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\GazeBase_v2_0\Fixation_Only\DVA\*.csv"
    const_gb = get_constants("GazeBase")
    gb_files = glob.glob(gb_folder)# get_files_with_pattern(gb_folder, const_gb['file_pattern'])
    for file in gb_files:
        data = pd.read_csv(file)
        removed_blink = remove_blink_gb(data, const_gb, 15)
        #remove_blink, const_gb = Interpolation.remove_blink_annot(data, const_roorda)
        new_cols = {const_gb['x_col']:'x', const_gb['y_col']:'y', const_gb['Annotations']:'flags'}
        removed_blink[const_gb['Annotations']] = 2
        b = Augmentation.flip_dataframe(removed_blink, const_gb).rename(columns=new_cols)
        c = Augmentation.reverse_data(removed_blink,const_gb).rename(columns=new_cols)
        folderpath = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\GazeBase"
        removed_blink = removed_blink.rename(columns=new_cols)
        removed_blink.to_csv(fr"{folderpath}\{Path(file).stem}.csv")
        #a.to_csv(fr"{folderpath}\{Path(file).stem}_reversed.csv")
        #b.to_csv(fr"{folderpath}\{Path(file).stem}_flipped.csv")
        #c.to_csv(fr"{folderpath}\{Path(file).stem}_reversed.csv")
    print(removed_blink.columns, b.columns, c.columns)
    print("done")

def GB_to_arcmin(input_folder):
    const_gb = get_constants('GazeBase')
    gb_files = get_csv_files_in_folder(input_folder)
    for file in gb_files:
        try:
            data = pd.read_csv(file)
            # Multiply with 60 to convert from DVA to Arcmin
            data, const_gb = Interpolation.remove_blink_annot(data, const_gb)
            data, const_gb = Interpolation.remove_sacc_annot(data, const_gb)
            data['x'] *= 60
            data['y'] *= 60
            file_name = Path(f"{Path(file).parents[1]}/Arcmin/{Path(file).stem}_arcmin.csv")
            data.to_csv(file_name, index=False)
        except Exception as e:
            print(e)

def get_best_HD(folderpath_to_jsons,variable, compare_json_file, normalize_01, output_folder=None):
    simulation_dict={}
    json_files = get_json_file(folderpath_to_jsons)
    with open(compare_json_file,'r') as compare:
        compare_data = json.load(compare)[variable]
    for file in json_files:
        with open(file, 'r') as open_file:
            data = json.load(open_file)[variable]
        file_name = Path(file).stem.replace('_n=50','').split(',')
        sim_rate = file_name[0].split('=')[1]
        cells_per_deg=file_name[1].split('=')[1]
        relaxation_rate = file_name[2].split('=')[1]
        hc=file_name[3].split('=')[1]
        HD = Evaluation.Evaluation.normalized_histogram_difference(data, compare_data, num_bins= 50, normalize_01=normalize_01)

        name = f'simulation rate={sim_rate}, cells per degree={cells_per_deg}, relaxation rate={relaxation_rate}, h_crit={hc}'
        simulation_dict.update({name: HD})
    # Sammle alle HD-Werte aus dem Dictionary
    hd_werte = list(simulation_dict.values())
    # Sortiere die HD-Werte in aufsteigender Reihenfolge
    hd_werte.sort()
    schwellenwert_index = int(0.3 * len(hd_werte))
    schwellenwert = hd_werte[schwellenwert_index]
    # Durchsuche das Dictionary nach Namen mit HD-Werten kleiner oder gleich dem Schwellenwert
    namen_mit_gueltigen_hd = [(name,hd) for name, hd in simulation_dict.items() if hd <= schwellenwert]
    columns= ['simulation rate','cells per degree','relaxation rate','h_crit', f'Mittelwert_{variable}', f'Median_{variable}', f'sigma_{variable}', 'HD']
    my_list=[]
    for element in namen_mit_gueltigen_hd:

        split_element = element[0].split(',')
        simrate = split_element[0].split('=')[1]
        cells =split_element[1].split('=')[1]
        relax =split_element[2].split('=')[1]
        hc = split_element[3].split('=')[1]
        file_name = rf'{folderpath_to_jsons}\AmpDurNum_simrate={simrate},Cells={cells},relaxationrate={relax},hc={hc}_n=50.json'
        hd = round(element[1], 3)
        with open(file_name, 'r') as open_file2:
            data2 = json.load(open_file2)[variable]
        mean = np.mean(data2)
        median = np.median(data2)
        std_dev = np.std(data2)
        my_list.append([simrate,cells,relax,hc,mean,median,std_dev,hd])
        file_name_new = f'HD={hd}_simulation_rate={simrate}_CellsPerDegree={cells}_RelaxationRate={relax}_HCrit={hc}.json'
        new_file_path = f"{Path(folderpath_to_jsons)}/{output_folder}/{file_name_new}"


        shutil.copyfile(Path(file_name), new_file_path)
    df = pd.DataFrame(my_list, columns=columns)
    excel_datei = rf"{folderpath_to_jsons}\ParameterInput_IntermicDur_SimDur=30s.xlsx"
    df.to_excel(excel_datei, index=False)
    csv_datei = rf"{folderpath_to_jsons}\ParameterInput_IntermicDur_SimDur=30s.csv"
    df.to_csv(csv_datei, index=False)


def merge_excel(file1_path, file2_path, output_path):
    """
    Merge two Excel files with identical column names and save the merged data as a new Excel file.

    Parameters:
    - file1_path (str): The file path to the first Excel file.
    - file2_path (str): The file path to the second Excel file.
    - output_path (str): The file path where the merged Excel data will be saved.

    Returns:
    - None

    """
    try:
        # Lese beide Excel-Dateien in Pandas DataFrames ein
        df1 = pd.read_excel(file1_path)
        df2 = pd.read_excel(file2_path)

        # Überprüfe, ob die beiden DataFrames die gleiche Anzahl an Spalten und identische Spaltennamen haben
        if len(df1.columns) != len(df2.columns) or list(df1.columns) != list(df2.columns):
            raise ValueError("Die beiden Excel-Dateien haben unterschiedliche Spaltenstrukturen.")

        # Füge die Daten der zweiten Datei an die erste an
        merged_df = pd.concat([df1, df2], ignore_index=True)

        # Speichere das gemergte DataFrame als Excel-Datei
        merged_df.to_excel(f'{output_path}.xlsx', index=False)
        merged_df.to_csv(f'{output_path}.csv', index=False)
    except Exception as e:
        print(f"Fehler beim Mergen der Excel-Dateien: {str(e)}")

def create_histogram_dual_w_HD(compare_filepath, file1, file2, feature, xlabel, compare_label, label1, label2,title=None):
    with open(
            compare_filepath,
            'r') as comp:
        compare_data = json.load(comp)[feature]
    with open(file1, 'r') as comp2:
        varlist2_1 = json.load(comp2)[feature]
    with open(file2, 'r') as comp3:
        varlist2_2 = json.load(comp3)[feature]
    hist_bins = 50
    savefigpath = file1
    savefig = f"{Path(savefigpath).parent}/DualHistogramLog_intermicsac.jpeg"
    Evaluation.hist_subplot_w_histdiff_log(compare_data, varlist2_1, compare_label, label1, savefig, xlabel,
                                           range_limits=(0, 2.5), normalize_01=False, title=title)
    Evaluation.dual_hist_subplot_w_histdiff_log(compare_data, varlist2_1, varlist2_2, compare_label, label1, label2, savefig, xlabel, range_limits=(0, 2.5), normalize_01=False)
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
def get_micsac_ang(folderpath, const_dict, ms_flag, drift_flag):
    files = []
    for filename in os.listdir(folderpath):
        if filename.endswith(".csv"):
            file_path = os.path.join(folderpath, filename)
            files.append(file_path)
    angles = []
    for file_path in files:
        df = pd.read_csv(file_path)
        onset_indices = []
        offset_indices = []
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
            print(f"Fehler beim Verarbeiten der Datei {file_path}: {e}")
            continue
    return angles
def plot_amplitude_hist(jsonpath):
    with open(jsonpath, 'r') as fp:
        data = json.load(fp)['MicsacAmplitudes']
    Vis.plot_prob_dist(data, "Histogramm der Amplituden von Mikrosakkaden bei Blickfeldgröße 2°", "Amplitude in Grad [dva]")

if __name__ == '__main__':
    const_roorda = get_constants('Roorda')
    const_gb = get_constants('GazeBase')
    roorda_folder = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    roorda_files = glob.glob(os.path.join(roorda_folder, "*.csv"))
    roorda_files = roorda_files[:-1]# get_files_with_pattern(gb_folder, const_gb['file_pattern'])

    for file in roorda_files:
        data = pd.read_csv(file)
        removed_blink, _ = Interpolation.remove_blink_annot(data,const_roorda, cutoff=0.015)

        new_cols = {const_roorda['x_col']: 'x', const_roorda['y_col']: 'y', const_roorda['Annotations']: 'flags'}
        removed_blink.loc[removed_blink[const_roorda['Annotations']] == 2, const_roorda['Annotations']] = 0
        b = Augmentation.flip_dataframe(removed_blink, const_roorda).rename(columns=new_cols)
        c = Augmentation.reverse_data(removed_blink, const_roorda).rename(columns=new_cols)
        folderpath = r"C:\Users\uvuik\Desktop\NewRoordaTrainingset"
        removed_blink = removed_blink.rename(columns=new_cols)
        removed_blink.to_csv(fr"{folderpath}\{Path(file).stem}.csv", index=False)
        #a.to_csv(fr"{folderpath}\{Path(file).stem}_reversed.csv")
        b.to_csv(fr"{folderpath}\{Path(file).stem}_flipped.csv",index=False)
        c.to_csv(fr"{folderpath}\{Path(file).stem}_reversed.csv",index=False)

    print("done")
    #Plotting FFTs of Datasets:

    data_roorda = pd.read_csv(r"C:\Users\uvuik\Desktop\NewRoordaTrainingset\20109R_003.csv")
    #data_roorda, const_roorda = Interpolation.remove_blink_annot(data_roorda,const_roorda)
    const_roorda['ValScaling'] = 1
    const_roorda['x_col'] = 'x'
    const_roorda['y_col'] = 'y'

    Vis.plot_xy(data_roorda, const_dict=const_roorda, xlim=(0,6), ylim = (-15, 15), ylabel='Position in Bogenminuten [arcmin]', savepath=r"C:\Users\uvuik\bwSyncShare\Bilder", filename="TestRoorda22")

    data_gb = pd.read_csv(r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\Roorda\20075L_003.csv")
    fft, fftfreq = Filt.fft_transform(data_gb, const_gb, 'x_col', 'y_col')
    Vis.plot_fft(fft, fftfreq, r"C:\Users\uvuik\Desktop\FrequencyAnalysis\GazeBaseBlinkStitchedFFT.jpeg", title = "Fourier-Analyse einer FEM-Trajektorie\n des GazeBase-Datensatz")


    #label_micsac(r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\GazeBase",const_dict=const_gb, mindur=6, vfac=10)
    file20s= r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Evaluation20s\BestHD_IntermicDur\HD=4.103_simulation_rate=100_CellsPerDegree=10_RelaxationRate=0.1_HCrit=6.9.json"
    file30s = r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Evaluation30s\BestHD_IntermicDur\HD=4.74_simulation_rate=100_CellsPerDegree=10_RelaxationRate=0.1_HCrit=6.9.json"
    create_histogram_dual_w_HD(file20s, file30s, file20s, 'IntermicDur', 'Intermikrosakkadische Intervalldauer in Sekunden [s]', compare_label='t_sim=20s',label1='t_sim=30s', label2='test',title= 'Histogramm der IMSI der Parameterkombination \n(f_sim=100Hz, L=21, epsilon=0.1, h_crit=6.9) bei 20s und 30s Simulationsdauer' )
    roorda_folder = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    roorda_files = get_files_with_pattern(roorda_folder, const_roorda['file_pattern'])
    i=0
    test_folder = r"C:\Users\uvuik\Desktop\TestFolderDataAugmentation"
    folderpath = r"C:\Users\uvuik\Desktop\TestFolderDataAugmentation"

    folderpath = r""

    create_histogram_dual_w_HD(
        r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\MicsacFeatures.json",
        r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Evaluation20s\BestHD_IntermicDur\HD=3.969_simulation_rate=150_CellsPerDegree=10_RelaxationRate=0.085_HCrit=8.9.json",
        r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Evaluation20s\BestHD_IntermicDur\HD=4.103_simulation_rate=100_CellsPerDegree=10_RelaxationRate=0.1_HCrit=6.9.json",
        'IntermicDur', 'Intermikrosakkadische Intervalldauer in Sekunden [s]', label1='(f_sim=150Hz, L=21, epsilon=0.085, h_crit=8.9)', label2='(f_sim=100Hz, L=21, epsilon=0.1, h_crit=6.9)')
    #Functions for plotting Histograms
    create_histogram_w_HD(
        r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\MicsacFeatures.json",
        r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Evaluation30s\BestHD_IntermicDur",
        'IntermicDur', 'Intermikrosakkadischer Abstand in Sekunden [s]', 'Roorda Lab', 'Math. Modell')
    get_best_HD(
        r"C:\Users\uvuik\Desktop\TestOrdner40s",
        'IntermicDur',
        r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\MicsacFeatures.json",
        normalize_01=False, output_folder="BestHD_IntermicDur")

    json_path = ""
    evaluate_intermic_hist(json_path, 2.5)
    Evaluation.Evaluation.generate_histogram_with_logfit(folderpath, range=(0, 2.5), compare_to_roorda=True)

    const_dict = ""
    data = ""
    drift_only_wo_micsac = EventDetection.drift_only_wo_micsac(data, const_dict)
    drift_only = EventDetection.drift_only(data, const_dict)
    drift_interpolated = EventDetection.drift_interpolated(data, const_dict)
    tremor_only = EventDetection.tremor_only(data, const_dict)
    filtered_drift = EventDetection.drift_only(data, const_dict)
    micsac_only = EventDetection.micsac_only(data, const_dict)
    #Evaluate all
    all_micsac_amp = []
    all_intermicsac = []
    all_micsac_dur = []
    all_micsac_vel = []
    all_drift_amp = []
    all_drift_vel = []
    all_tremor_amp = []
    all_tremor_vel = []
    roorda_files = ""
    for file in roorda_files:
        data = pd.read_csv(file)
        #Get Resolution
        x = data['xx']
        y = data['yy']
        a_x = x.sort_values()
        a_y = y.sort_values()
        diff_x = a_x.diff()
        diff_y = a_y.diff()
        nn_x = diff_x[diff_x!=0.0]
        nn_y = diff_y[diff_y.notnull()]


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

    with open(r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\Statistics(mindur=10,vfac=21).csv", 'w',
         newline='') as csvfile:
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


    #Vis.plot_prob_dist([i*60 for i in all_tremor_amp], 'Amplitude of tremor', 'Amplitude in arcsec')#Why does this lead to an error?
    #Mikrosakkaden
    Vis.plot_prob_dist(all_micsac_amp, 'Amplituden der Mikrosakkaden', 'Amplitude in Bogenminuten')
    Vis.plot_prob_dist(all_intermicsac, 'Intermikrosakkadische Intervalldauer', 'Zeit in s')
    Vis.plot_prob_dist(all_micsac_vel, 'Geschwindigkeit von Mikrosakkaden', 'Geschwindigkeit in °/s')
    Vis.plot_prob_dist(all_micsac_dur, 'Dauer von Mikrosakkaden', 'Zeit in s')
    #Drift
    Vis.plot_prob_dist(all_drift_amp, 'Amplituden des Drifts', 'Amplitude in Bogenminuten')
    Vis.plot_prob_dist(all_drift_vel, 'Geschwindigkeit des Drifts', 'Geschwindigkeit in Bogenminuten/s')
    #Tremor
    Vis.plot_prob_dist(all_tremor_amp, 'Amplituden des Tremors', 'Amplitude in Bogenminuten')
    Vis.plot_prob_dist(all_tremor_vel, 'Geschwindigkeit des Tremors', 'Geschwindigkeit in Bogenminuten/s')




    amp_drift, vel_drift = Evaluation.get_drift_statistics(blink_rm, const_dict)
    amp_tremor, vel_tremor = Evaluation.get_tremor_statistics(blink_rm, const_dict)
    fft, fftfreq = Filt.fft_transform(filtered_drift, const_roorda, 'x_col')
    Vis.plot_fft(fft, fftfreq)
    Vis.plot_xy(filtered_drift, const_roorda)

    # Working with Roorda_data
    # roorda_folder = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    folder = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    df, micsac = detect_all_micsac(folder, mindur=10, vfac=21, const_dict=const_roorda,resample=False, save=True)
    # Visualize both:
    # Vis.plot_xy(data, const_dict)
    # Vis.plot_xy(data, const_dict, colors=['red', 'orange'], labels=['x Roorda', 'y Roorda'])
    # Vis.plot_xy(gab_data, const_gb, colors=['blue', 'violet'], labels=['x GazeBase', 'y GazeBase'])

    #Microsaccades according to Roorda:
    roorda_micsac = Microsaccades.get_roorda_micsac(data)
    Vis.print_microsacc(data, const_roorda, roorda_micsac)

    '''
    # Filtering Bandpass
    Drift_c = EventDetection.filter_drift(cubic, const_dict)
    Tremor_c = EventDetection.filter_tremor(cubic, const_dict)
    Drift_p = EventDetection.filter_drift(piece_poly, const_dict)
    Tremor_p = EventDetection.filter_tremor(piece_poly, const_dict)
    Drift = EventDetection.filter_drift(spliced, const_dict)
    Tremor = EventDetection.filter_tremor(spliced, const_dict)
    '''
