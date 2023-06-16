#Filter for Drift
filtered_drift = EventDetection.drift_only(roorda_data, const_roorda)

#Calculate and plot frequency domain
fft, fftfreq = Filt.fft_transform(filtered_drift, const_roorda, 'x_col')
Vis.plot_fft(fft, fftfreq)