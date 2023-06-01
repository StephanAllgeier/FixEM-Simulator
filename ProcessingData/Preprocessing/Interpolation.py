import copy

import numpy as np
import scipy


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
    def remove_blink(df, const_dict, time_cutoff): #time_cutoff in ms
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
        data_frame = data_frame.reset_index()
        data_frame[const_dict['time_col']] = data_frame.index / const_dict['f']
''        return data_frame
