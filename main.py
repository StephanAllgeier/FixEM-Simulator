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
    def __init__(self, dataset_name):
        if dataset_name == "Roorda":
            self.Name = "Roorda"
            self.f = 1920
            self.x_col = 'xx'
            self.y_col = 'yy'
            self.time_col = 'TimeAxis'
            self.ValScaling = 1 / 60
            self.TimeScaling = 1
            self.BlinkID = 3
            self.Annotations = 'Flags'
            self.file_pattern = "\d{5}[A-Za-z]_\d{3}\.csv"
            self.rm_blink = False
        elif dataset_name == "GazeBase":
            self.Name = "GazeBase"
            self.f = 1000
            self.x_col = 'x'
            self.y_col = 'y'
            self.time_col = 'n'
            self.ValScaling = 1
            self.TimeScaling = 1 / 1000
            self.BlinkID = -1
            self.Annotations = 'lab'
            self.rm_blink = False
            self.SakkID = 2  # Einheiten für y-Koordinate sind in dva (degrees of vision angle)
        elif dataset_name == "GazeBase_arcmin":
            self.Name = "GazeBase"
            self.f = 1000
            self.x_col = 'x'
            self.y_col = 'y'
            self.time_col = 'n'
            self.ValScaling = 1/60
            self.TimeScaling = 1 / 1000
            self.BlinkID = -1
            self.Annotations = 'lab'
            self.rm_blink = False
            self.SakkID = 2  
        elif dataset_name == "OwnData":
            self.Name = "OwnData"
            self.f = 500
            self.x_col = 'x'
            self.y_col = 'y'
            self.time_col = 'Time'
            self.ValScaling = 1
            self.TimeScaling = 1
            self.BlinkID = None
            self.Annotations = 'lab'
            self.rm_blink = False

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
        detected_micsac[Path(file).stem] = len(micsac_annot)  # (micsac_detec[0])
    if save:
        print('Evaluation done')
        data_name = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\MicSacDetected.xlsx"
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
def Evaluate_intermic_hist(json_path):
    Evaluation.Evaluation.generate_histogram_with_logfit(json_path, range = (0,4))

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
        b = Augmentation.flip_dataframe(removed_blink, const_gb).rename(columns=new_cols)
        c = Augmentation.reverse_data(removed_blink,const_gb).rename(columns=new_cols)
        folderpath = r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\GazeBase"
        removed_blink = removed_blink.rename(columns=new_cols)
        removed_blink.to_csv(fr"{folderpath}\{Path(file).stem}.csv")
        #a.to_csv(fr"{folderpath}\{Path(file).stem}_reversed.csv")
        b.to_csv(fr"{folderpath}\{Path(file).stem}_flipped.csv")
        c.to_csv(fr"{folderpath}\{Path(file).stem}_reversed.csv")
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
        file_name = rf'C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\TIMEOUT_0.015s\AmpDurNum_simrate={simrate},Cells={cells},relaxationrate={relax},hc={hc}_n=50.json'
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
    excel_datei = r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Timeout_0.015s\ParameterInput_IntermicDur_0.3-0.9.xlsx"
    df.to_excel(excel_datei, index=False)
    csv_datei = r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Timeout_0.015s\ParameterInput_IntermicDur_0.3-0.9.csv"
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

def create_histogram_dual_w_HD(compare_filepath, file1, file2, feature, xlabel):
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
    Evaluation.hist_subplot_w_histdiff_log(compare_data, varlist2_1, 'Roorda Lab', '(f_sim=200Hz, L=51, epsilon=0.08, h_crit=5.4)', savefig, xlabel, range_limits=(0, 2.5), normalize_01=False)
    Evaluation.dual_hist_subplot_w_histdiff_log(compare_data, varlist2_1, varlist2_2, 'Roorda Lab', '(f_sim=150Hz, L=21, epsilon=0.085, h_crit=8.9)', '(f_sim=100Hz, L=21, epsilon=0.1, h_crit=6.9)', savefig, xlabel, range_limits=(0, 2.5), normalize_01=False)
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
if __name__ == '__main__':
    augmentation()
    #roorda_file=    r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\10003L_004.csv"
    #data = pd.read_csv(roorda_file)
    #const_roorda = get_constants('Roorda')
    #data, const_roorda = Interpolation.remove_blink_annot(data, const_roorda)

    #const_gb = get_constants('GazeBase')
    #gb_test_file = pd.read_csv(r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\GazeBase_v2_0\Fixation_Only\DVA\S_1004_S1_FXS.csv")
    #gb_file, const_gb = Interpolation.remove_blink_annot(gb_test_file, const_gb)
    #gb_file, const_gb = Interpolation.remove_sacc_annot(gb_file, const_gb)
    #fft, fftfreq = Filtering.fft_transform(gb_file, const_gb, 'x_col')
    #Vis.plot_fft(fft, fftfreq)
    #file = r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Timeout_0.015s\BestHD_IntermicDur\HD=3.969_simulation_rate=150_CellsPerDegree=10_RelaxationRate=0.085_HCrit=8.9.json"
    #with open(file, 'r') as fp:
    #    data = json.load(fp)['MicsacAmplitudes']
    #Vis.plot_prob_dist(data, "Histogramm der Amplituden von Mikrosakkaden bei Blickfeldgröße 2°", "Amplitude in Grad [dva]")

    #merge_excel(r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\Alt\test1.xlsx", r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\Alt\test2.xlsx",r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\Alt\test3")
    #jsonfiles= get_json_file(r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\TestOrdner GuiInput")
    #roorda_folder = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    #const_roorda = get_constants('Roorda')
    #speicherpfad_roorda = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\MicsacFeatures.json"
    #create_histogram_w_HD(
    #    r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\MicsacFeatures.json",
    #    r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Timeout_0.015s\BestHD_IntermicDur",
    #    'IntermicDur', 'Intermikrosakkadischer Abstand in Sekunden [s]', 'Roorda Lab', 'Math. Modell')
    #get_best_HD(
    #    r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Timeout_0.015s",
    #    'IntermicDur',
    #    r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\MicsacFeatures.json",
    #    normalize_01=False, output_folder="BestHD_IntermicDur")
    create_histogram_dual_w_HD(
        r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\MicsacFeatures.json",
        r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Timeout_0.015s\BestHD_IntermicDur\HD=4.989_simulation_rate=200_CellsPerDegree=25_RelaxationRate=0.08_HCrit=5.4.json",
        r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Timeout_0.015s\BestHD_IntermicDur\HD=4.103_simulation_rate=100_CellsPerDegree=10_RelaxationRate=0.1_HCrit=6.9.json",'IntermicDur', 'Intermikrosakkadische Intervalldauer in Sekunden [s]')



    create_histogram_w_HD(
        r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\MicsacFeatures.json",
        r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Timeout_0.015s\BestHD_IntermicDur",
        'IntermicDur', 'Intermikrosakkadische Intervalldauer in Sekunden [s]', 'Roorda Lab', 'Math. Modell')




    Evaluation.Evaluation.generate_histogram_with_logfit(folderpath, range=(0, 2.5), compare_to_roorda=True)

    GB_to_arcmin(
        r"E:\GazeBase Dataset\DVA")

    gb_test_file = pd.read_csv(r"C:\Users\fanzl\bwSyncShare\Documents\GazeBase_v2_0\Fixation_Only\S_1002_S1_FXS.csv")
    gb_file, const_gb = Interpolation.remove_blink_annot(gb_test_file, const_gb)
    #gb_file, const_gb = Interpolation.remove_sacc_annot(gb_file, const_gb)
   #gb_file, const_gb = Interpolation.dva_to_arcmin(gb_file, const_gb)

    roorda_test_file = pd.read_csv(
        r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\20109R_003.csv")
    const_roorda = get_constants("Roorda")
    own_file = r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\NeueKombinationen\dur=30.0_cells=25.0_SamplingF=500.0_SimulationF=150.0_relaxationr=0.1\Bestes\7_Signal_NumMS=21.csv"
    own_data = pd.read_csv(own_file)
    own_dict = get_constants('OwnData')
    save_path = r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\NeueKombinationen\dur=30.0_cells=25.0_SamplingF=500.0_SimulationF=150.0_relaxationr=0.1\Bestes\Vergleich Simulation Roorda"
    Vis.plot_xy_dual(own_data, roorda_test_file, own_dict, const_roorda, 'Simulation', 'Roorda Lab', savepath=save_path, t_on=10, t_off=13, title='Augenbewegungen in x- und y-Koordinaten')
    json_path = r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\NeueKombinationen\AmpDurNum_simrate=150,Cells=25,relaxationrate=0.1,hc=3.9_n=50.json"
    Evaluate_intermic_hist(json_path)
    roorda_folder =r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    speicherpfad_roorda = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    const_roorda = get_constants('Roorda')
    Evaluation.Evaluation.get_intermic_hist_dataset_rooda(folderpath=roorda_folder, speicherpfad=speicherpfad_roorda, const_dict=const_roorda)

    # OWN DATA
    #folder_path= r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\hc_3,4,10-30s, n=50\Traces AmpDurNum_simrate=100,Cells=25,relaxationrate=0.09,hc=3.4\dur=30.0_cells=25.0_SamplingF=500.0_SimulationF=100.0_relaxationr=0.09"
    # micsac_dur = read_all_values_from_csv(r"C:\Users\fanzl\PycharmProjects\MasterarbeitIAI\Test1\Test\dur=10.0_cells=25.0\SamplingF=1000.0_SimulationF=200.0", '*intermic_dur.csv', 'Intermicsac Duration [s]')
    # micsac_amp = read_all_values_from_csv(r"C:\Users\fanzl\PycharmProjects\MasterarbeitIAI\Test1\Test\dur=10.0_cells=25.0\SamplingF=1000.0_SimulationF=200.0", '*micsac_amp.csv', 'Micsac Amplitude [deg]')
    #all_own_files = get_csv_files_in_folder(folder_path)
    own_file =pd.read_csv(r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\hc_3,4,10-30s, n=50\dur=30.0_cells=25.0_SamplingF=500.0_SimulationF=100.0_relaxationr=0.09\2_Signal_NumMS=17.csv")
    const_own = get_constants('OwnData')
    roorda_test_file =  pd.read_csv(r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\20109R_003.csv")
    const_roorda = get_constants("Roorda")
    const_gb = get_constants('GazeBase')
    gb_test_file = pd.read_csv(r"C:\Users\fanzl\bwSyncShare\Documents\GazeBase_v2_0\Fixation_Only\S_1006_S1_FXS.csv")
    gb_file, const_gb = Interpolation.remove_blink_annot(gb_test_file, const_gb)
    gb_file , const_gb = Interpolation.remove_sacc_annot(gb_file, const_gb)
    Vis.plot_xy_dual(own_file, gb_file, const_own, const_gb, 'Eigene Daten', 'GazeBase', color1=None, color2=None, t_on=10, t_off=14)
    '''
    for file in all_own_files:

        own_data = pd.read_csv(file)
        x = own_data['x']
        y = own_data['y']
        f = 500
        t = own_data['Time']
        own_dict =get_constants('OwnData')
        plt.plot(t, x, label='x', color='g')
        plt.plot(t, y, label='y', color='b')
        plt.xlabel('Zeit in s')
        plt.ylabel('Auslenkung in Grad [°]')
        plt.title('Test1')
        plt.legend()
        plt.show()
        Vis.plot_xy_trace(own_dict)
    own_micsacs = Microsaccades.find_micsac(own_data, own_dict)
    roorda_data = pd.read_csv(roorda_test_file)
    const_roorda = get_constants("Roorda")
    roorda_files = get_files_with_pattern(roorda_folder, const_roorda['file_pattern'])
    '''
    #allgeier_folder = r"\\os.lsdf.kit.edu\iai-projects\iai-aida\Daten_Allgeier\fixational_eye_movement_trajectories"
    
    # gb_file = r"C:\Users\fanzl\bwSyncShare\Documents\GazeBase_v2_0\Fixation_Only\S_1006_S1_FXS.csv"
    '''
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

    #Vis.plot_xy(blink_rm[0:round(6000/1000*const_roorda['f'])],const_roorda,color=['b','orange'], labels=['x-Achse', 'y-Achse'], title='Menschliche Augenbewegung während der Fixation')
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
'''
    # with open(r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\Statistics(mindur=10,vfac=21).csv", 'w',
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







    gb_data = pd.read_csv(gb_file)

    Vis.plot_xy_trace(roorda_data[0:8000], const_roorda, label='Eye trace', color='orange')
    #data = gb_data
    #const_dict = const_gb
    '''
    # Working with Roorda_data
    # roorda_folder = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    # df, micsac = detect_all_micsac(roorda_folder, mindur=10, vfac=21, const_dict=const_roorda,resample=False, save=True)
    # Visualize both:
    # Vis.plot_xy(data, const_dict)
    # Vis.plot_xy(data, const_dict, colors=['red', 'orange'], labels=['x Roorda', 'y Roorda'])
    # Vis.plot_xy(gab_data, const_gb, colors=['blue', 'violet'], labels=['x GazeBase', 'y GazeBase'])

    # Microsaccades according to Roorda:    roorda_micsac = Microsaccades.get_roorda_micsac(roorda_data)
    # Vis.print_microsacc(roorda_data, const_roorda, roorda_micsac)
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
    '''
#    Vis.print_microsacc(spliced, const_dict, micsacc)
