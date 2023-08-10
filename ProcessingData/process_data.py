import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import visualize_xyt as vis


def detect_msacc(data, samplingrate, width, threshold):
    time = data['TimeAxis']
    x_vals = data['xx']
    y_vals = data['yy']
    msacc = [(0, 0)]
    t = data['TimeAxis']
    if len(x_vals) != len(y_vals):
        return
    for i in range(len(x_vals) - (width - 1)):
        win_x = x_vals[i:i + width]
        win_y = y_vals[i:i + width]
        if isinstance(win_x, pd.Series):
            win_x = win_x.tolist()
            win_y = win_y.tolist()
        x_first, x_last = win_x[0], win_x[-1]
        y_first, y_last = win_y[0], win_y[-1]
        x_diff, y_diff = abs(x_last - x_first), abs(y_last - y_first)
        if np.sqrt((np.square(x_diff) + np.square(y_diff))) > threshold:
            # If it is within an alreaady existing window, discard
            if i < msacc[-1][1]:
                continue
            msacc.append((i, i + 15))

    msacc.pop(0)
    return msacc


def interpolate():
    # Adding a statement
    # adding second statement
    return


def plot_w_msacc(x_vals, y_vals, t, msacc):
    vis.plot_xy(x_vals, y_vals, t)
    t = t.tolist()
    for i in msacc:
        plt.axvline(x=t[i[0]])
        plt.axvline(x=t[i[1]])


@staticmethod


if __name__ == '__main__':
    file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\10003L_002.csv"
    data = pd.read_csv(file)
    x_vals = data['xx']
    y_vals = data['yy']
    t = data['TimeAxis']
    msacc = detect_msacc(data, 1000, 15, 15)
    plot_w_msacc(x_vals, y_vals, t, msacc)

    print('horray')
