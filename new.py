import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs

af_csv = glob.glob('mimic_perform_af_csv/*.csv')
non_af_csv = glob.glob('mimic_perform_non_af_csv/*.csv')

column_names = [
    "Time",
    "PPG",
    "ECG",
    "resp"
]

af_dfs = []
non_af_dfs = []

df = pd.read_csv('mimic_perform_af_csv/mimic_perform_af_010_data.csv')

ecg = df.ECG
plt.plot(ecg)

wander_baseline = pd.Series(sps.medfilt(ecg, kernel_size=61))
plt.plot(wander_baseline)

# flatten ecg
flat_ecg = ecg - wander_baseline
plt.plot(flat_ecg)
plt.title('ecg')
plt.show()

# filter ecg w butterworth lowpass
Wn = 0.2
b, a = sps.butter(4, Wn, 'low', analog=False)
filt_ecg = sps.filtfilt(b, a, ecg)
flat_filt_ecg = filt_ecg - wander_baseline
r_waves,_ = sps.find_peaks(flat_filt_ecg, height=0.2)

plt.plot(filt_ecg)
plt.plot(flat_filt_ecg)
plt.plot(r_waves, flat_filt_ecg[r_waves], "x")
plt.title('flat_filt_ecg')
plt.show()

d_ecg, peaks_d_ecg = lab_funcs.decg_peaks(flat_filt_ecg, df.Time)
filt_peaks = lab_funcs.d_ecg_peaks(d_ecg, peaks_d_ecg, df.Time, 0.5, 0.5)
Rpeaks = lab_funcs.Rwave_peaks(flat_filt_ecg, d_ecg, filt_peaks, df.Time)
Rpeak_intervals = np.diff(Rpeaks)
avg_intv = np.mean(Rpeak_intervals)
print(avg_intv)