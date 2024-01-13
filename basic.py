

import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import glob
import lab_funcs

def Rpeak_intervals(df):
    # filters
    Wn = 0.2
    b, a = scipy.signal.butter(4, Wn, 'low', analog=False)
    filt_ecg = scipy.signal.filtfilt(b, a, df.ECG)
    # plt.plot(df.ECG)
    # plt.plot(filt_ecg)
    # plt.title('filt_ecg')
    # plt.show()

    d_ecg, peaks_d_ecg = lab_funcs.decg_peaks(filt_ecg, df.Time)
    filt_peaks = lab_funcs.d_ecg_peaks(d_ecg, peaks_d_ecg, df.Time, 0.5, 0.5)
    Rpeaks = lab_funcs.Rwave_peaks(filt_ecg, d_ecg, filt_peaks, df.Time)
    Rpeak_intervals = np.diff(Rpeaks)
    hr = 1/Rpeak_intervals*60
    # plt.show()
    # plt.plot(Rpeak_intervals, label='Peakint')
    # plt.plot(hr, label='hr')
    # plt.show()
    return Rpeak_intervals


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

for csv in af_csv:
    df = pd.read_csv(csv)
    peak_intv = Rpeak_intervals(df)
    avg_intv = np.mean(peak_intv)
    print(avg_intv)
    af_dfs.append(avg_intv)
plt.plot(af_dfs)

# for csv in non_af_csv:
#     df = pd.read_csv(csv)
#     peak_intv = Rpeak_intervals(df)
#     avg_intv = np.mean(peak_intv)
#     print(avg_intv)
#     non_af_dfs.append(avg_intv)


df = pd.read_csv('mimic_perform_af_csv/mimic_perform_af_005_data.csv')
# peaks,_ = scipy.signal.find_peaks(d.ECG)
# plt.plot(d.Time, d.ECG, label='ECG')
# plt.show()
# Wn = 0.2
# b, a = scipy.signal.butter(4, Wn, 'low', analog=False)
# print(a, b)
# filt_ecg = scipy.signal.filtfilt(b, a, df.ECG)
# peaks,_ = scipy.signal.find_peaks(df.ECG)
# plt.plot(df.Time, df.ECG)
# plt.plot(df.Time[peaks], df.ECG[peaks], "x")
# plt.show()
# plt.plot(filt_ecg)
# plt.title('filt_ecg')
# plt.show()
# Rpeak_intervals(d)