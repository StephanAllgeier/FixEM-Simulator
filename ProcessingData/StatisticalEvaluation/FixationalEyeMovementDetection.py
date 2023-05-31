import copy

import numpy as np
import scipy
import pandas as pd
from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox import microsac_detection
from ProcessingData.Preprocessing.Filtering import Filtering


class EventDetection():
    @staticmethod
    def find_micsacc(df, constant_dict, mindur=3, vfac=5):
        '''
        parameters:
        df=dataframe to work with
        constant_dict = dictionary belongig to the dataset with information about the structure of the dataset
        coordinate = which coordinate to evaluate
        '''
        input_array = df[[constant_dict['x_col'], constant_dict['y_col']]].to_numpy()
        micsacc = microsac_detection.microsacc(input_array, sampling=constant_dict['f'], mindur=mindur, vfac=vfac)
        return micsacc

    # -------------------------------------------------------------------------------------------------------------------
    # Drift and Tremor, thus filtered signal necessary
    # Tremor: f =[50Hz, 100Hz]
    @staticmethod
    def filter_tremor(df, constant_dict):
        lowcut = 50
        highcut = 100
        return_frame = copy.deepcopy(df)
        return_frame[constant_dict['x_col']] = Filtering.butter_bandpass_filter(df[constant_dict['x_col']],
                                                                                lowcut=lowcut, highcut=highcut,
                                                                                fs=constant_dict['f'])
        #Filtered=Filtering.butter_bandpass_filter(spliced, lowcut=50, highcut=100, fs=1920, order=5)
        #Filtered_2= pd.DataFrame(Filtered, columns=['index'].extend(list(roorda_data.columns)))