import numpy as np
from matplotlib import pyplot as plt


def plot_xy_dual(dataset1, dataset2, const_dict1, const_dict2, color1=None, color2=None, labels1=None, labels2=None,
                 title='Augenbewegungen in x- und y-Koordinaten'):
    if labels1 is None:
        labels1 = ['x', 'y']
    if labels2 is None:
        labels2 = ['x', 'y']
    if color1 is None:
        color1 = ['blue', 'orange']
    if color2 is None:
        color2 = ['blue', 'orange']

    f1 = const_dict1['f']
    t1 = dataset1[const_dict1['time_col']] * const_dict1['TimeScaling']
    x1 = dataset1[const_dict1['x_col']] * const_dict1['ValScaling']
    y1 = dataset1[const_dict1['y_col']] * const_dict1['ValScaling']

    f2 = const_dict2['f']
    t2 = dataset2[const_dict2['time_col']]
    x2 = dataset2[const_dict2['x_col']] * const_dict2['ValScaling']
    y2 = dataset2[const_dict2['y_col']] * const_dict2['ValScaling']

    mask1 = (t1 >= 2) & (t1 <= 6)
    mask2 = (t2 >= 2) & (t2 <= 6)
    t1 = np.array(t1[mask1])
    x1 = np.array(x1[mask1])
    y1 = np.array(y1[mask1])
    t2 = np.array(t2[mask2])
    x2 = np.array(x2[mask2])
    y2 = np.array(y2[mask2])
    plt.figure(figsize=(6, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t1, x1, label=labels1[0], color=color1[0])
    plt.plot(t1, y1, label=labels1[1], color=color1[1])
    plt.xlim(2, 6)
    plt.ylim(min(min(x1), min(x2), min(y1), min(y2)), max(max(x1), max(y1), max(x2), max(y2)))
    plt.xlabel('Zeit in s')
    plt.ylabel('Position in Bogenminuten [arcmin]')
    plt.title(title + ' - Roorda Lab Datensatz')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t2, x2, label=labels2[0], color=color2[0])
    plt.plot(t2, y2, label=labels2[1], color=color2[1])
    plt.xlim(2, 6)
    plt.ylim(min(min(x1), min(x2), min(y1), min(y2)), max(max(x1), max(y1), max(x2), max(y2)))
    plt.xlabel('Zeit in s')
    plt.ylabel('Position in Bogenminuten [arcmin]')
    plt.title(title + ' - Gazebase')
    plt.legend()

    save_path = r"C:\Users\fanzl\bwSyncShare\Documents\Texte\Masterarbeit_AnzlingerFabian\Bilder\DualPlot_GB_ROODA\Comparison_GB_ROORDA"
    plt.savefig(save_path + '.svg', format='svg', dpi=600)
    plt.savefig(save_path + '.jpeg', format='jpeg', dpi=600)
    plt.savefig(save_path + '.pdf', format='pdf')
    plt.show()
