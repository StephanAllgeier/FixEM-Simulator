import matplotlib.pyplot as plt
import pandas as pd


class Visualize():
    plt.rcParams['lines.linewidth'] = 1

    @staticmethod
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

    @staticmethod
    def plot_xy_trace(dataset):
        x = dataset['x_vals']
        y = dataset['y_vals']
        plt.plot(x, y)
        plt.xlabel('X in arcminutes')
        plt.ylabel('Y in arcminutes')
        plt.title('Trace of Eye Movement')
        plt.legend()
        plt.show()

