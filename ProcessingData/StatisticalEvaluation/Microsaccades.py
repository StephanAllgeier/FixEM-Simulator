import matplotlib.pyplot as plt
import numpy as np

from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox import microsac_detection
from ProcessingData.StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection


class Microsaccades():

    @staticmethod
    def count_micsac_annot(df):
        '''
        return number of annotated microsaccades
        '''
        micsac = Microsaccades.get_roorda_micsac(df)
        count = len(micsac)
        return count

    @staticmethod
    def find_micsac(df, constant_dict, mindur=3, vfac=5):
        '''
        parameters:
        df=dataframe to work with, units of the traces is in degrees of visual angle
        constant_dict = dictionary belongig to the dataset with information about the structure of the dataset
        coordinate = which coordinate to evaluate
        returns tuple with tuple[0] = microsaccades
        '''
        #Filtering Signal like in Paper "Eye Movement Analysis in Simple Visual Tasks"
        df = EventDetection.filter_drift(df, constant_dict=constant_dict, highcut=40, order=5)
        input_array = df[[constant_dict['x_col'], constant_dict['y_col']]].to_numpy()
        micsac = microsac_detection.microsacc(input_array, sampling=constant_dict['f'], mindur=mindur, vfac=vfac)
        return micsac

    @staticmethod
    def get_roorda_micsac(df):
        # Input is dataframe from Roorda_Database. It Returns list of lists containing onset and offset of microsaccades
        mic_sac_idx = df[df['Flags'] == 1].index
        current_sublist = []
        indexes = []
        for i in range(len(mic_sac_idx)):
            if i == 0 or mic_sac_idx[i] != mic_sac_idx[i - 1] + 1:
                if current_sublist:
                    indexes.append(current_sublist)
                current_sublist = [mic_sac_idx[i]]
            else:
                current_sublist.append(mic_sac_idx[i])

        # Füge die letzte Teil-Liste hinzu, falls vorhanden
        if current_sublist:
            indexes.append(current_sublist)
        micsac_onoff = []
        for liste in indexes:
            micsac_onoff.append([liste[0], liste[-1]])
        return micsac_onoff