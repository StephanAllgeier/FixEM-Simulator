import pandas as pd
import numpy as np
import scipy
from Preprocessing.Interpolation import Interpolation
from Visualize import Visualize

def get_constants(dataset_name):
    if dataset_name == "Roorda":
        return {"f": 1920, "x_col": 'xx', "y_col": 'yy', "time_col": 'TimeAxis', "ValScaling" : 1, "TimeScaling" : 1}
    elif dataset_name == "GazeBase":
        return {"f": 1000, "x_col": 'x', "y_col": 'y', "time_col": 'n', "ValScaling" : 60, "TimeScaing" : 1/1000} #Einheiten für y-Kooridnate ist in dva (degrees of vision angle)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #roorda_files = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    roorda_test_file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\10003L_001.csv"
    roorda_data = pd.read_csv(roorda_test_file)
    const_roorda = get_constants("Roorda")

    gazebase_file = r"E:\GazeBase\GazeBase_v2_0\Fixation_Only\S_1001_S1_FXS.csv"
    gazebase_data = pd.read_csv(gazebase_file)
    const_gazebase = get_constants("GazeBase")
    #
    #Visualize both:
    Visualize.plot_xy(roorda_data, const_roorda)
    #Visualize.plot_xy(dataset_gazebase)

    #Interpolation
    a = Interpolation.interp_cubic(roorda_data, const_roorda)

    print('Done')
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
