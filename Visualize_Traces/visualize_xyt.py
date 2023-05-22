import matplotlib.pyplot as plt
import pandas as pd


def plot_xy(x, y, t):
    plt.plot(t, x, label='x')
    plt.plot(t, y, label='y')
    plt.xlabel('Time')
    plt.ylabel('Position in arcmin')
    plt.title('Position over Time')
    plt.legend()
    plt.show()

def plot_xy_trace(x,y):
    plt.plot(x, y)
    plt.xlabel('X in arcminutes')
    plt.ylabel('Y in arcminutes')
    plt.title('Trace of Eye Movement')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    file = r"C:\Users\fanzl\bwSyncShare\Documents\Dataset\External\EyeMotionTraces_Roorda Vision Berkeley\10003L_001.csv"
    data = pd.read_csv(file)
    time = data['TimeAxis']
    x_vals = data['xx']
    y_vals = data['yy']
    plot_xy(x_vals, y_vals, time)
    plot_xy_trace(x_vals, y_vals)
    print('Done')
