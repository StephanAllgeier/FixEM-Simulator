import pandas as pd
import numpy as np
import scipy

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file = r"E:\GazeBase\GazeBase_v2_0\Fixation_Only\S_1001_S1_FXS.csv"
    data = pd.read_csv(file)
    gazebase = {"f": 1000, "x_vals": data['x'], "y_vals": data['y'], "time": data['n']}
    # roorda = {"f": 1920, "x_vals": data['xx'], "y_vals": data['yy'], "time": data['TimeStamp']}

    plot_xy(gazebase)
    plot_xy_trace(gazebase)
    print('Done')
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
