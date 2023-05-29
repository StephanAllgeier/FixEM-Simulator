import numpy as np
import scipy


class Interpolation():
    @staticmethod
    def interp_moncub(df, const_dict):
        df[const_dict['y_col']] = scipy.interpolate.PchipInterpolator(df[const_dict['time_col']],
                                                                      df[const_dict['y_col']], axis=0)
        df[const_dict['x_col']] = scipy.interpolate.PchipInterpolator(df[const_dict['time_col']],
                                                                      df[const_dict['x_col']], axis=0)
        return df

    @staticmethod
    def interp_cubic(df, const_dict):
        df[const_dict['x_col']] = df[const_dict['x_col']].interpolate(method='pchip')
        '''
        scipy.interpolate.CubicSpline(df[const_dict['time_col']],
                                                                df[const_dict['x_col']].apply(lambda x: float()),
                                                                axis=0).apply(lambda x: float())
        df[const_dict['y_col']] = (scipy.interpolate.CubicSpline(df[const_dict['time_col']],
                                                                df[const_dict['y_col']].apply(lambda x: float()), axis=0)).apply(lambda x: float())
        '''
        return df