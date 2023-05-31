import pandas as pd
import numpy as np
import scipy
from Preprocessing.Interpolation import Interpolation
from Visualize import Visualize
from Preprocessing.Filtering import Filtering
from StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection

def get_constants(dataset_name):
    if dataset_name == "Roorda":
        return {"f": 1920, "x_col": 'xx', "y_col": 'yy', "time_col": 'TimeAxis', "ValScaling" : 1, "TimeScaling" : 1}
    elif dataset_name == "GazeBase":
        return {"f": 1000, "x_col": 'x', "y_col": 'y', "time_col": 'n', "ValScaling" : 60, "TimeScaling" : 1/1000} #Einheiten für y-Kooridnate ist in dva (degrees of vision angle)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #roorda_files = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    roorda_test_file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\10003L_001.csv"
    roorda_data = pd.read_csv(roorda_test_file)
    const_roorda = get_constants("Roorda")

    gazebase_file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\Testing\S_1001_S1_FXS.csv"
    gazebase_data = pd.read_csv(gazebase_file)
    const_gazebase = get_constants("GazeBase")
    #

    #Visualize both:
    #Visualize.plot_xy(roorda_data, const_roorda)
    #Visualize.plot_xy(gazebase_data, const_gazebase)
    #Visualize.plot_xy(dataset_gazebase)
    #Visualize.plot_xy(roorda_data, const_roorda, color=['red','orange'], labels=['x Roorda','y Roorda'])
    #Visualize.plot_xy(gazebase_data, const_gazebase, color=['blue','violet'], labels=['x GazeBase','y GazeBase'])

    #Interpolation
    cubic = Interpolation.interp_cubic(roorda_data, const_roorda)
    piece_poly = Interpolation.interp_monocub(roorda_data, const_roorda)
    spliced = Interpolation.splice_together(roorda_data, const_roorda)

    #FastFourierTransformation
    fft, fftfreq = Filtering.fft_transform(Interpolation.splice_together(gazebase_data, const_gazebase), const_gazebase, 'x_col')
    Visualize.plot_fft(fft, fftfreq)
    #Filtering Bandpass


    micsacc = EventDetection.find_micsacc(spliced, const_roorda, mindur=12)

    Visualize.print_microsacc(spliced, const_roorda, micsacc)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
