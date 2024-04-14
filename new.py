import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs
import preprocess_ecg
import scipy.io.wavfile

af_csv = glob.glob('mimic_perform_af_csv/*.csv')
non_af_csv = glob.glob('mimic_perform_non_af_csv/*.csv')

column_names = [
    "Time",
    "PPG",
    "ECG",
    "resp"
]

sample_rate = 120

af_dfs = []
non_af_dfs = []

df = pd.read_csv('mimic_perform_af_csv/mimic_perform_af_005_data.csv')

start = 0
end = -1
ecg = df.ECG[start:end]
ecg = preprocess_ecg.interpolate(ecg, start)
time = df.Time[start:end]
# plt.plot(ecg)

# Apply a 40 Hz low-pass filter to the original data
filtered_lp = preprocess_ecg.lowpass(ecg, 40, sample_rate)
# Apply 5 Hz high-pass filter
filtered_hp = preprocess_ecg.highpass(filtered_lp, 5, sample_rate)
# Apply 5-40 Hz band-pass filter
filtered_bp = preprocess_ecg.bandpass(filtered_hp, [5, 40], sample_rate)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 3), sharex=True, sharey=True)
ax1.plot(ecg)
ax1.set_title("Original Signal")
ax1.margins(0, .1)
ax1.grid(alpha=.5, ls='--')
ax2.plot(filtered_lp)
ax2.set_title("Low-Pass Filter (40 Hz)")
ax2.grid(alpha=.5, ls='--')
ax3.plot(filtered_hp)
ax3.set_title("High-Pass Filter (5 Hz)")
ax3.grid(alpha=.5, ls='--')
ax4.plot(filtered_bp)
ax4.set_title("Band-Pass Filter (5-40 Hz)")
ax4.grid(alpha=.5, ls='--')
plt.tight_layout()
plt.show()

# d_ecg, peaks_d_ecg = lab_funcs.decg_peaks(filtered_bp, time)
# filt_peaks = lab_funcs.d_ecg_peaks(d_ecg, peaks_d_ecg, time, 0.5, 0.5)
# Rpeaks = lab_funcs.Rwave_peaks(filtered_bp, d_ecg, filt_peaks, time)
# Rpeak_intervals = np.diff(Rpeaks)
# avg_intv = np.mean(Rpeak_intervals)
# print(avg_intv)

preprocess_ecg.Rpeak_intervals(filtered_bp, time)