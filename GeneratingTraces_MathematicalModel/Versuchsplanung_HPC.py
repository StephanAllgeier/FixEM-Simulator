import itertools
import json
import statistics

import pandas as pd
import openpyxl
import datetime

from GeneratingTraces_MathematicalModel import RandomWalkBased

def generate_combinations(lists):
    return list(itertools.product(*lists))

def Versuchsplanung():
    simulations_rates = [100, 150, 200, 250]
    cells_per_degree = [10, 25, 50, 100, 150]
    relaxation_rates = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
    all_combinations = generate_combinations([simulations_rates, cells_per_degree, relaxation_rates])
    folder_names = ["hc=7,4"]
    n = 10
    headers = ["simulation rate", "cells per degree", "relaxation rate", 'h_crit', "mean intermicsac duration [s]",
               "median intermicsac duration [s]", "stdev intermicsac duration [s]", "mean micsac amp [deg]",
               "median micsac amp [deg]", "stdev micsac amp [deg]", "Number of Micsac - Mean",
               "Number of Micsac - Median"]
    all_data=[]
    foldernum=0
    for folder in folder_names:
        foldernum+=1
        j = 0
        all_part_data = []
        for combination in all_combinations:
            j += 1
            print(
                f"Combination Nr. {j*foldernum}/{len(all_combinations)*len(folder_names)}: {combination} [simulations rate, cells per degree, relaxation rate], {folder}")
            comb_micsac_amp = []
            comb_intermic_dur = []
            comb_micsacs = []
            for i in range(n):
                simulation_rate = combination[0]
                cells_per_deg = combination[1]
                relaxation_rate = combination[2]
                hc = float(folder[-3:].replace(',', '.'))
                folderpath = None
                micsac_amp, intermic_duration, num_of_micsac = RandomWalkBased.RandomWalk.randomWalk(
                    simulation_freq=simulation_rate, potential_resolution=cells_per_deg,
                    relaxation_rate=relaxation_rate, hc=hc, folderpath=folderpath, duration=30, sampling_frequency=500,
                    number_id=i)
                comb_micsac_amp.extend(micsac_amp)
                comb_intermic_dur.extend(intermic_duration)
                comb_micsacs.extend([num_of_micsac])
            # Statistical_evaluation:
            if not len(comb_micsac_amp) < 2 and not len(comb_intermic_dur) < 2:
                mean_intermicsac_dur = statistics.mean(comb_intermic_dur)
                median_intermicsac_dur = statistics.median(comb_intermic_dur)
                sigma_intermicsac_dur = statistics.stdev(comb_intermic_dur)
                mean_micsac_amp = statistics.mean(comb_micsac_amp)
                median_micsac_amp = statistics.median(comb_micsac_amp)
                sigma_intermicsac_amp = statistics.stdev(comb_micsac_amp)
                micsacs_mean = statistics.mean(comb_micsacs)
                micsacs_median = statistics.median(comb_micsacs)
            else:
                mean_intermicsac_dur, median_intermicsac_dur, sigma_intermicsac_dur, mean_micsac_amp, median_micsac_amp, sigma_intermicsac_amp = 0, 0, 0, 0, 0, 0
            data = [simulation_rate, cells_per_deg, relaxation_rate, hc, mean_intermicsac_dur, median_intermicsac_dur,
                    sigma_intermicsac_dur, mean_micsac_amp, median_micsac_amp, sigma_intermicsac_amp, micsacs_mean,
                    micsacs_median]
            all_data.append(data)
            all_part_data.append(data)
            print(f"{datetime.datetime.now()}: {j * foldernum / len(all_combinations) / len(folder_names)}% done")
        partially_df = pd.DataFrame(all_part_data, columns=headers)
        filename_excel_partially = f"MicsacStatistics30s10, {folder}.xlsx"
        filename_csv_partially = f"MicsacStatistics30s10, {folder}.csv"
        partially_df.to_excel(rf"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Teiltabellen\0.4 vals\{filename_excel_partially}", index=False)
        partially_df.to_csv(rf"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\Teiltabellen\0.4 vals\{filename_csv_partially}", index=False)

    df = pd.DataFrame(all_data, columns=headers)
    file_name_excel = "MicsacStatistics30s25_,4vals.xlsx"
    file_name_csv = "MicsacStatistics30s25_,4vals.csv"
    df.to_excel(rf"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\{file_name_excel}",
                index=False)
    df.to_csv(rf"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\{file_name_csv}",
              index=False)
    print(len(all_combinations))

def save_combination_evaluation_to_json():
    float_options = [(20, 10, 0.1, 1.9), (50, 25, 0.002, 1.9), (50, 10, 0.001, 3.4), (50, 10, 0.002, 3.4),
                     (50, 10, 0.01, 3.4), (50, 10, 0.1, 3.4), (50, 10, 0.001, 3.9), (50, 10, 0.005, 3.9),
                     (50, 10, 0.01, 3.9), (100, 10, 0.001, 5.4), (100, 10, 0.002, 5.4), (100, 10, 0.01, 5.4),
                     (100, 10, 0.05, 5.4), (100, 10, 0.001, 5.9), (100, 10, 0.005, 5.9), (100, 10, 0.05, 6.9)]
    drop_var = ['simulation_freq', "potential_resolution", "relaxation_rate", "hc"]
    n = 100
    headers = ["simulation rate", "cells per degree", "relaxation rate", 'h_crit', "mean intermicsac duration [s]",
               "median intermicsac duration [s]", "stdev intermicsac duration [s]", "mean micsac amp [deg]",
               "median micsac amp [deg]", "stdev micsac amp [deg]", "Number of Micsac - Mean",
               "Number of Micsac - Median"]
    all_data = []
    for combination in float_options:
        comb_micsac_amp = []
        comb_intermic_dur = []
        comb_micsacs = []
        for i in range(n):
            simulation_rate = combination[0]
            cells_per_deg = combination[1]
            relaxation_rate = combination[2]
            hc = combination[3]
            micsac_amp, intermic_duration, num_of_micsac = RandomWalkBased.RandomWalk.randomWalk(
                simulation_freq=simulation_rate, potential_resolution=cells_per_deg, relaxation_rate=relaxation_rate,
                hc=hc, duration=30, sampling_frequency=500, number_id=i)
            comb_micsac_amp.extend(micsac_amp)
            comb_intermic_dur.extend(intermic_duration)
            comb_micsacs.extend([num_of_micsac])
        my_dict = {"MicsacAmplitudes": comb_micsac_amp, "IntermicDur": comb_intermic_dur, "NumMicsac": comb_micsacs}
        with open(
                fr'C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\30s\AmpDurNum_simrate={str(simulation_rate)},Cells={str(cells_per_deg)},relaxationrate={str(relaxation_rate)},hc={str(hc)}.json',
                'w') as fp:
            json.dump(my_dict, fp, indent=4)
def get_combination_from_excel(excel_file):
    wb = openpyxl.load_workbook(excel_file)
    sheet = wb.active
    headers = [cell.value for cell in sheet[1]][0:4]
    all_combinations = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        tupel1 = dict(zip(headers, row[0:4]))
        all_combinations.append(tupel1)
    wb.close()
    return all_combinations
def Versuchsplanung_excelcombinations(excel_file=None):
    if isinstance(excel_file, None):
        excel_file = r"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\10-30s, n=50, neu\Filter.xlsx"
    # Open Excelfile
    combinations = get_combination_from_excel(excel_file)
    j = 0
    headers = ["simulation rate", "cells per degree", "relaxation rate", 'h_crit', "mean intermicsac duration [s]",
               "median intermicsac duration [s]", "stdev intermicsac duration [s]", "mean micsac amp [deg]",
               "median micsac amp [deg]", "stdev micsac amp [deg]", "Number of Micsac - Mean",
               "Number of Micsac - Median"]
    all_data = []
    for combination in combinations:
        j += 1
        print(
            f"Combination Nr. {j}/{len(combinations)}: {combination} [simulations rate, cells per degree, relaxation rate, hc]")
        comb_micsac_amp = []
        comb_intermic_dur = []
        comb_micsacs = []
        n = 50
        for i in range(n):
            simulation_rate = combination['simulation rate']
            cells_per_deg = combination['cells per degree']
            relaxation_rate = combination['relaxation rate']
            hc = combination['h_crit']
            folderpath = rf'C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell'
            micsac_amp, intermic_duration, num_of_micsac = RandomWalkBased.RandomWalk.randomWalk(
                simulation_freq=simulation_rate, potential_resolution=cells_per_deg,
                relaxation_rate=relaxation_rate, hc=hc, folderpath=folderpath, duration=30, sampling_frequency=500,
                number_id=i)
            comb_micsac_amp.extend(micsac_amp)
            comb_intermic_dur.extend(intermic_duration)
            comb_micsacs.extend([num_of_micsac])
        # Statistical_evaluation:
        if not len(comb_micsac_amp) < 2 and not len(comb_intermic_dur) < 2:
            mean_intermicsac_dur = statistics.mean(comb_intermic_dur)
            median_intermicsac_dur = statistics.median(comb_intermic_dur)
            sigma_intermicsac_dur = statistics.stdev(comb_intermic_dur)
            mean_micsac_amp = statistics.mean(comb_micsac_amp)
            median_micsac_amp = statistics.median(comb_micsac_amp)
            sigma_intermicsac_amp = statistics.stdev(comb_micsac_amp)
            micsacs_mean = statistics.mean(comb_micsacs)
            micsacs_median = statistics.median(comb_micsacs)
        else:
            mean_intermicsac_dur, median_intermicsac_dur, sigma_intermicsac_dur, mean_micsac_amp, median_micsac_amp, sigma_intermicsac_amp = 0, 0, 0, 0, 0, 0
        data = [simulation_rate, cells_per_deg, relaxation_rate, hc, mean_intermicsac_dur, median_intermicsac_dur,
                sigma_intermicsac_dur, mean_micsac_amp, median_micsac_amp, sigma_intermicsac_amp, micsacs_mean,
                micsacs_median]
        all_data.append(data)
    df = pd.DataFrame(all_data, columns=headers)
    file_name_excel = "MicsacStatistics_new.xlsx"
    file_name_csv = "MicsacStatistics_new.csv"
    df.to_excel(rf"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\{file_name_excel}",
                index=False)
    df.to_csv(rf"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\{file_name_csv}",
              index=False)

if __name__ == '__main__':
    Versuchsplanung()
    print('Statement')


