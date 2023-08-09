import itertools
import statistics

import pandas as pd

from GeneratingTraces_MathematicalModel import RandomWalkBased

def generate_combinations(lists):
    return list(itertools.product(*lists))

if __name__ == '__main__':
    simulations_rates = [10,20, 50,100,150,200,250]
    cells_per_degree = [10,25,50,100,150,200,300,360]
    relaxation_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    all_combinations = generate_combinations([simulations_rates, cells_per_degree, relaxation_rates])
    folder_names = ["hc=3,4", "hc=7,4", "hc=7,9"]
    n=25
    headers=["simulation rate", "cells per degree", "relaxation rate", 'h_crit', "mean intermicsac duration [s]", "median intermicsac duration [s]", "stdev intermicsac duration [s]", "mean micsac amp [deg]", "median micsac amp [deg]", "stdev micsac amp [deg]", "Number of Micsac - Mean", "Number of Micsac - Median"]
    all_data = []
    for folder in folder_names:
        j=0
        for combination in all_combinations:
            j+=1
            print(f"Combination Nr. {j}/{len(all_combinations)}: {combination} [simulations rate, cells per degree, relaxation rate]")
            comb_micsac_amp = []
            comb_intermic_dur=[]
            comb_micsacs = []
            for i in range(n):
                simulation_rate = combination[0]
                cells_per_deg = combination[1]
                relaxation_rate = combination[2]
                hc = float(folder[-3:].replace(',','.'))
                folderpath = rf'C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\{folder}'
                micsac_amp, intermic_duration, num_of_micsac = RandomWalkBased.RandomWalk.randomWalk(simulation_freq=simulation_rate, potential_resolution=cells_per_deg, relaxation_rate=relaxation_rate, hc = hc, folderpath=folderpath, duration=10, sampling_frequency=500,  number_id=i)
                comb_micsac_amp.extend(micsac_amp)
                comb_intermic_dur.extend(intermic_duration)
                comb_micsacs.extend([num_of_micsac])
            #Statistical_evaluation:
            if not len(comb_micsac_amp) <2 and not len(comb_intermic_dur) <2:
                mean_intermicsac_dur = statistics.mean(comb_intermic_dur)
                median_intermicsac_dur = statistics.median(comb_intermic_dur)
                sigma_intermicsac_dur = statistics.stdev(comb_intermic_dur)
                mean_micsac_amp = statistics.mean(comb_micsac_amp)
                median_micsac_amp = statistics.median(comb_micsac_amp)
                sigma_intermicsac_amp = statistics.stdev(comb_micsac_amp)
                micsacs_mean = statistics.mean(comb_micsacs)
                micsacs_median = statistics.median(comb_micsacs)
            else:
                mean_intermicsac_dur, median_intermicsac_dur, sigma_intermicsac_dur, mean_micsac_amp, median_micsac_amp, sigma_intermicsac_amp = 0,0,0,0,0,0
            data = [simulation_rate, cells_per_deg, relaxation_rate, hc, mean_intermicsac_dur, median_intermicsac_dur, sigma_intermicsac_dur, mean_micsac_amp, median_micsac_amp, sigma_intermicsac_amp, micsacs_mean, micsacs_median]
            all_data.append(data)
            print(f"{j / len(all_combinations) / len(folder_names)}% done")
    df = pd.DataFrame(all_data, columns=headers)
    file_name_excel= "MicsacStatistics[hc=3,4].xlsx"
    file_name_csv = "MicsacStatistics[hc=3,4].csv"
    df.to_excel(rf"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\{file_name_excel}", index=False)
    df.to_csv(rf"C:\Users\uvuik\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\{file_name_csv}", index=False)
    print(len(all_combinations))
