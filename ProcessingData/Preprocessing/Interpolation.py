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