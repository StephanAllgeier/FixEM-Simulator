import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['lines.linewidth'] = 1
def plot_xy(dataset):
    f = dataset['f']
    t = dataset['time']
    x = dataset['x_vals']
    y = dataset['y_vals']
    plt.plot(t/f, x, label='x')
    plt.plot(t/f, y, label='y')
    plt.xlabel('Time')
    plt.ylabel('Position in arcmin')
    plt.title('Position over Time')
    plt.legend()
    plt.show()

def plot_xy_trace(dataset):
    x = dataset['x_vals']
    y = dataset['y_vals']
    plt.plot(x, y)
    plt.xlabel('X in arcminutes')
    plt.ylabel('Y in arcminutes')
    plt.title('Trace of Eye Movement')
    plt.legend()
    plt.show()




if __name__ == '__main__':

    file = r"E:\GazeBase\GazeBase_v2_0\Fixation_Only\S_1001_S1_FXS.csv"
    data = pd.read_csv(file)
    gazebase = {"f": 1000, "x_vals": data['x'], "y_vals": data['y'], "time": data['n']}
    #roorda = {"f": 1920, "x_vals": data['xx'], "y_vals": data['yy'], "time": data['TimeStamp']}

    plot_xy(gazebase)
    plot_xy_trace(gazebase)
    print('Done')
