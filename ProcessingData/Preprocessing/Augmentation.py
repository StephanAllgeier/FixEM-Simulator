import numpy as np
import pandas as pd
from scipy import signal


class Augmentation():
    @staticmethod
    def add_gaussian_noise(dataframe, dataframe_dict, mean=0, std=0.01):
        """
            Fügt gaussisches Rauschen zur 'x'- und 'y'-Spalte des DataFrames hinzu.

            Args:
                dataframe (pd.DataFrame): Der DataFrame mit den Spalten 't', 'x' und 'y'.
                mean (float): Mittelwert der Normalverteilung für das Rauschen.
                std (float): Standardabweichung der Normalverteilung für das Rauschen.

            Returns:
                pd.DataFrame: Der ursprüngliche DataFrame mit hinzugefügtem Rauschen.
            """
        noisy_data = dataframe.copy()
        x_col = dataframe_dict['x_col']
        y_col = dataframe_dict['y_col']
        noise = np.random.normal(mean, std, size=len(noisy_data))
        noisy_data[x_col] += noise
        noisy_data[y_col] += noise
        return noisy_data

    @staticmethod
    def reverse_data(dataframe, dataframe_dict):
        """
            Kehrt die Reihenfolge der Spalten im DataFrame außer der ersten ("Time") um.

            Args:
                dataframe (pd.DataFrame): Der DataFrame, der umgekehrt werden soll.

            Returns:
                pd.DataFrame: Der umgekehrte DataFrame mit unveränderten Zeitwerten.
            """

        reversed_data = dataframe.copy()
        time_column_index = dataframe.columns.get_loc(dataframe_dict['time_col'])

        # Wende die Umkehrung auf die ausgewählten Spalten an
        reversed_data.iloc[:, time_column_index + 1:] = reversed_data.iloc[:, time_column_index + 1:].values[::-1]
        return reversed_data

    @staticmethod
    def slice_df(dataframe, const_dict, segment_length):
        """
           Teilt den DataFrame in Unter-DataFrames von jeweils {segment_length} Sekunden Länge auf.

           Args:
               dataframe (pd.DataFrame): Der DataFrame, der aufgeteilt werden soll.
                                         Er muss eine Spalte "Zeit" haben.
               f (float): Die Abtastfrequenz des DataFrames.

           Returns:
               List[pd.DataFrame]: Eine Liste von Unter-DataFrames mit jeweils {segment_length} Sekunden Länge.
           """
        f = const_dict['f']
        samples_per_segment = int(f*segment_length)
        segments = []
        start_idx = 0

        while start_idx+samples_per_segment <= len(dataframe):
            end_idx = start_idx + samples_per_segment
            segment = dataframe.iloc[start_idx:end_idx]
            segments.append(segment.values)
            start_idx = end_idx
        return segments
    @staticmethod
    def flip_dataframe(dataframe, const_dict):
        """
            Ändert das Vorzeichen der Spalten "x" und "y" im DataFrame.

            Args:
                dataframe (pd.DataFrame): Der DataFrame, in dem das Vorzeichen geändert werden soll.

            Returns:
                pd.DataFrame: Der modifizierte DataFrame mit geänderten Vorzeichen in den Spalten "x" und "y".
            """
        modified_data = dataframe.copy()
        x_col = const_dict['x_col']
        y_col = const_dict['y_col']
        modified_data[[x_col, y_col]] = -modified_data[[x_col, y_col]]
        return modified_data

    @staticmethod
    def resample(df, const_dict, f_target=1000):
        '''
        resample a signal from original frequency fs to target frequency frs
        '''
        interm_frame = df[[const_dict['time_col'], const_dict['x_col'], const_dict['y_col']]]
        fs = const_dict['f']
        resampling_ratio = f_target / fs
        num_output_samples = int(len(interm_frame) * resampling_ratio)

        return_x = pd.Series(signal.resample(interm_frame[const_dict['x_col']], num_output_samples),
                             name=const_dict['x_col'])
        return_y = pd.Series(signal.resample(interm_frame[const_dict['y_col']], num_output_samples),
                             name=const_dict['y_col'])
        return_t = pd.Series(np.linspace(0, df[const_dict['time_col']].iloc[-1], num_output_samples),
                             name=const_dict['time_col'])

        # Using pydsm
        # return_x = pydsm.resample(interm_frame[const_dict['x_col']], fs, f_target)
        # return_y = pydsm.resample(interm_frame[const_dict['y_col']], fs, f_target)
        # return_t = pydsm.resample(interm_frame[const_dict['t_col']], fs, f_target)
        const_dict['f'] = f_target
        return pd.concat([return_t, return_x, return_y], axis=1), const_dict