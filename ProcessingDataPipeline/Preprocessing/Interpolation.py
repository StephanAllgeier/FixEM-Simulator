import numpy as np
import scipy


class Interpolation():
    @staticmethod
    def interpolate_cubic(df, x_column, y_column):
        df[y_column] = scipy.interpolate.PchipInterpolator(df[x_column], df[y_column], axis=0)

