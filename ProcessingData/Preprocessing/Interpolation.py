import copy

import numpy as np
import pandas as pd
import scipy
from scipy import signal


class Interpolation():
    @staticmethod
    def interp_monocub(df, const_dict):
        data_frame = copy.deepcopy(df)
        data_frame[const_dict['x_col']] = data_frame[const_dict['x_col']].interpolate(method='pchip')
        data_frame[const_dict['y_col']] = data_frame[const_dict['y_col']].interpolate(method='pchip')
        return data_frame

    @staticmethod
    def interp_akima(df, const_dict):
        data_frame = copy.deepcopy(df)
        data_frame[const_dict['x_col']] = data_frame[const_dict['x_col']].interpolate(method='akima')
        data_frame[const_dict['y_col']] = data_frame[const_dict['y_col']].interpolate(method='akima')
        return data_frame

    @staticmethod
    def interp_cubic(df, const_dict):
        data_frame = copy.deepcopy(df)
        data_frame[const_dict['x_col']] = data_frame[const_dict['x_col']].interpolate(method='cubic')
        data_frame[const_dict['y_col']] = data_frame[const_dict['y_col']].interpolate(method='cubic')
        return data_frame

    @staticmethod
    def splice_together(df, const_dict):
        data_frame = df.dropna(subset=[const_dict['x_col'], const_dict['y_col']], how='any')
        data_frame = data_frame.reset_index()
        data_frame[const_dict['time_col']] = data_frame.index/const_dict['f']
        return data_frame

    @staticmethod
    def remove_blink(df, const_dict, time_cutoff, remove_start=False, remove_end = False,remove_start_time= 0, remove_end_time= 0): #time_cutoff in ms
        #This function cuts out blink movements
        data_frame = copy.deepcopy(df)

        x_nan = df[df[const_dict['x_col']].isnull()].index.to_list()
        y_nan = df[df[const_dict['y_col']].isnull()].index.to_list()
        if x_nan != y_nan:
            print('Die beiden Einträge sind ungleich lang, wird verworfen.')
        #Splitting lists into sublists
        indexes = []
        current_sublist = []
        padding_count = round(time_cutoff *const_dict['f'] //1000)
        for i in range(len(x_nan)):
            if i == 0 or x_nan[i] != x_nan[i - 1] + 1:
                if current_sublist:
                    padded_list = list(range(current_sublist[0] - padding_count, current_sublist[0])) + current_sublist + list(range(current_sublist[-1] + 1, current_sublist[-1] + padding_count + 1))
                    indexes.append(padded_list)
                current_sublist = [x_nan[i]]
            else:
                current_sublist.append(x_nan[i])

        # Füge die letzte Teil-Liste hinzu, falls vorhanden
        if current_sublist:
            indexes.append(current_sublist)

        #Delete Rows from Frame
        for liste in indexes:
            data_frame = data_frame.drop(liste)
        if remove_end:
            data_frame = data_frame[0:-remove_end_time*const_dict['f']//1000]
        if remove_start:
            data_frame= data_frame[remove_start_time*const_dict['f']//1000:]
        data_frame = data_frame.reset_index()
        data_frame[const_dict['time_col']] = data_frame.index / const_dict['f'] / const_dict['TimeScaling']
        return data_frame

    @staticmethod
    def remove_blink_annot(df, const_dict):
        data_frame = copy.deepcopy(df)
        blink_idx = df[df[const_dict['Annotations']] == const_dict['BlinkID']].index
        current_sublist = []
        indexes = []
        for i in range(len(blink_idx)):
            if i == 0 or blink_idx[i] != blink_idx[i - 1] + 1:
                if current_sublist:
                    indexes.append(current_sublist)
                current_sublist = [blink_idx[i]]
            else:
                current_sublist.append(blink_idx[i])

        # Füge die letzte Teil-Liste hinzu, falls vorhanden
        if current_sublist:
            indexes.append(current_sublist)
        for liste in indexes:
            data_frame = data_frame.drop(liste)
        data_frame = data_frame.reset_index()
        data_frame[const_dict['time_col']] = data_frame.index / const_dict['f']
        return data_frame

    @staticmethod
    def convert_arcmin_to_dva(df, const_dict):
        return_df= copy.deepcopy(df)
        factor = 1/60
        return_df[const_dict['x_col']], return_df[const_dict['y_col']] = return_df[const_dict['x_col']].multiply(factor), return_df[const_dict['y_col']].multiply(factor)
        return return_df

    @staticmethod
    def resample(df, const_dict, f_target=1000):
        '''
        resample a signal from original frequency fs to target frequency frs
        '''
        interm_frame = df[[const_dict['time_col'], const_dict['x_col'], const_dict['y_col']]]
        fs = const_dict['f']
        resampling_ratio = f_target/fs
        num_output_samples = int(len(interm_frame)*resampling_ratio)

        return_x = pd.Series(signal.resample(interm_frame[const_dict['x_col']], num_output_samples), name=const_dict['x_col'])
        return_y = pd.Series(signal.resample(interm_frame[const_dict['y_col']], num_output_samples), name=const_dict['y_col'])
        return_t = pd.Series(np.linspace(0, df[const_dict['time_col']].iloc[-1], num_output_samples), name=const_dict['time_col'])

        # Using pydsm
        # return_x = pydsm.resample(interm_frame[const_dict['x_col']], fs, f_target)
        # return_y = pydsm.resample(interm_frame[const_dict['y_col']], fs, f_target)
        # return_t = pydsm.resample(interm_frame[const_dict['t_col']], fs, f_target)
        const_dict['f'] = f_target
        return pd.concat([return_t, return_x, return_y], axis=1), const_dict
