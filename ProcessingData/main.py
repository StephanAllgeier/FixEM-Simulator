import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from Preprocessing.Interpolation import Interpolation
from Visualize import Visualize as Vis
from Preprocessing.Filtering import Filtering as Filt
from StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection


def get_constants(dataset_name):
    if dataset_name == "Roorda":
        return {"f": 1920, "x_col": 'xx', "y_col": 'yy', "time_col": 'TimeAxis', "ValScaling": 1, "TimeScaling": 1,
                'BlinkID': 3, 'Annotations': 'Flags'}
    elif dataset_name == "GazeBase":
        return {"f": 1000, "x_col": 'x', "y_col": 'y', "time_col": 'n', "ValScaling": 60,
                "TimeScaling": 1 / 1000, 'BlinkID': -1, 'Annotations': 'lab'}  # Einheiten für y-Kooridnate ist in dva (degrees of vision angle)

def get_events(df, const_dict, msac_mindur=4):
    return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # roorda_files = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley"
    roorda_test_file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\20109R_003.csv"
    roorda_data = pd.read_csv(roorda_test_file)
    const_roorda = get_constants("Roorda")

    gb_file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\Testing\S_1001_S1_FXS.csv"
    gb_data = pd.read_csv(gb_file)
    const_gb = get_constants("GazeBase")
    #
    data = gb_data
    const_dict = const_gb

    # Visualize both:
    #Vis.plot_xy(data, const_dict)
    #Vis.plot_xy(data, const_dict, colors=['red', 'orange'], labels=['x Roorda', 'y Roorda'])
    #Vis.plot_xy(gab_data, const_gb, colors=['blue', 'violet'], labels=['x GazeBase', 'y GazeBase'])

    #Microsaccades according to Roorda:
    roorda_micsac = Vis.get_roorda_micsac(roorda_data)
    Vis.print_microsacc(roorda_data, const_roorda, roorda_micsac)

    # Interpolation
    cubic = Interpolation.interp_cubic(data, const_dict)
    piece_poly = Interpolation.interp_monocub(data, const_dict)
    spliced = Interpolation.splice_together(data, const_dict)
    blink_removed = Interpolation.remove_blink(data, const_dict, 10, remove_start=True, remove_end = False,remove_start_time= 20, remove_end_time= 0)
    # FastFourierTransformation
    fft_spliced, fftfreq_spliced = Filt.fft_transform(spliced, const_dict, 'x_col')
    Vis.plot_fft(fft_spliced, fftfreq_spliced)

    # Filtering Bandpass
    Drift_c = EventDetection.filter_drift(cubic, const_dict)
    Tremor_c = EventDetection.filter_tremor(cubic, const_dict)
    Drift_p = EventDetection.filter_drift(piece_poly, const_dict)
    Tremor_p = EventDetection.filter_tremor(piece_poly, const_dict)
    Drift = EventDetection.filter_drift(spliced, const_dict)
    Tremor = EventDetection.filter_tremor(spliced, const_dict)

    #Plotting Tremor/Drift and belonging Frequencie Spectrum

    fft2, fftfreq2 = Filt.fft_transform(Drift, const_dict, 'x_col')
    Vis.plot_fft(fft2, fftfreq2)
    micsacc = EventDetection.find_micsacc(spliced, const_dict, mindur=12)

    Vis.print_microsacc(spliced, const_dict, micsacc)
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
