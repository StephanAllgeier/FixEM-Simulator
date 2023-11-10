import pandas as pd
import scipy.stats
import numpy as np
data = pd.read_csv(r"C:\Users\uvuik\Documents\Code\MasterarbeitIAI\GeneratingTraces_RGANtorch\FEM\GazeBaseLabels.csv")
drifts = np.array(list(data['drift']))
ms = np.array(list(data['ms']))
# Beispiel für Pearson's Korrelationskoeffizient
pearson_corr, _ = scipy.stats.pearsonr(drifts, ms)

# Beispiel für Spearman's Rangkorrelationskoeffizient
spearman_corr, _ = scipy.stats.spearmanr(drifts, ms)

# Beispiel für den Kolmogorov-Smirnov-Test
ks_statistic, ks_p_value = scipy.stats.ks_2samp(drifts, ms)
print(f'Folgende Statistiken wurden gefunden für die Länge der Drifts mit den zugehörigen Mikrosakkaden:\n\nPearson-Korrelation: {pearson_corr}\nSpearman-Korrelation: {spearman_corr}\nKolmogorov-Smirnov p-Value: {ks_p_value} ')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# Scatterplot erstellen
plt.scatter(drifts, ms, label='Datenpunkte')
plt.xlabel('Driftdauer in [ms]', fontsize=14)
plt.ylabel('Mikrosakkadendauer in [ms]', fontsize=14)
plt.title('Scatterplot von Driftdauer und Mikrosakkadendauer', fontsize=16)

# Berechne die Korrelationskoeffizienten
pearson_corr, _ = scipy.stats.pearsonr(drifts, ms)
spearman_corr, _ = scipy.stats.spearmanr(drifts, ms)

# Berechne den Kolmogorov-Smirnov-Test
ks_statistic, ks_p_value = scipy.stats.ks_2samp(drifts, ms)

# Linearer Fit
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(drifts, ms)
line = slope * drifts + intercept

# Füge den linearen Fit zum Plot hinzu
plt.plot(drifts, line, color='red', label=f'Linearer Fit (p-Wert: {p_value:.4f})')

# Füge Text mit den Korrelationskoeffizienten und dem p-Wert hinzu
text = f'Pearson-Korrelation: {pearson_corr:.4f}\nSpearman-Korrelation: {spearman_corr:.4f}\nKolmogorov-Smirnov p-Wert: {ks_p_value:.3f}'


# Zeige den Plot an
legend = plt.legend(loc='upper right')
plt.text(12500,150 , text, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'))

# Verschiebe die Legende an die gewünschte Position
plt.gca().add_artist(legend)

plt.tight_layout()
plt.savefig(fr"C:\Users\uvuik\bwSyncShare\Documents\ScatterplotGazeBaseDriftMS.jpg", dpi=600)