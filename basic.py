

import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs

def flatten_filter(ecg):
    wander_baseline = pd.Series(sps.medfilt(ecg, kernel_size=61))
    # plt.plot(wander_baseline)

    # filter ecg w butterworth lowpass
    Wn = 0.2
    b, a = sps.butter(4, Wn, 'low', analog=False)
    filt_ecg = sps.filtfilt(b, a, ecg)

    # flatten ecg
    flat_filt_ecg = filt_ecg - wander_baseline
    # plt.plot(df.ECG)
    # plt.plot(filt_ecg)
    # plt.title('filt_ecg')
    # plt.show()

    return flat_filt_ecg

# find intervals
def Rpeak_intervals(ecg, time):
    d_ecg, peaks_d_ecg = lab_funcs.decg_peaks(ecg, time)
    filt_peaks = lab_funcs.d_ecg_peaks(d_ecg, peaks_d_ecg, time, 0.5, 0.5)
    Rpeaks = lab_funcs.Rwave_peaks(ecg, d_ecg, filt_peaks, time)
    Rpeak_intervals = np.diff(Rpeaks)
    # plt.show()
    # plt.plot(Rpeak_intervals, label='Peakint')
    # plt.show()
    avg_intv = np.mean(Rpeak_intervals)
    # min_intv = np.min(Rpeak_intervals)
    # max_intv = np.max(Rpeak_intervals)
    # print(avg_intv, min_intv, max_intv)
    return Rpeak_intervals

def fft(ecg):
    # Compute the FFT
    fft_result = np.fft.fft(ecg)
    frequencies = np.fft.fftfreq(len(fft_result), 1/1000)  # Assuming a sampling frequency of 1000 Hz

    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(fft_result))
    plt.title('FFT of Filtered ECG Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()
    

# af_csv = glob.glob('mimic_perform_af_csv/*.csv')
# non_af_csv = glob.glob('mimic_perform_non_af_csv/*.csv')

# column_names = [
#     "Time",
#     "PPG",
#     "ECG",
#     "resp"
# ]

# af_dfs = []
# non_af_dfs = []

# for csv in af_csv:
#     df = pd.read_csv(csv)
#     ecg = flatten_filter(df.ECG)
#     peak_intv = Rpeak_intervals(ecg)
#     fft(ecg)
#     af_dfs.append(peak_intv)

# print('non')
# for csv in non_af_csv:
#     df = pd.read_csv(csv)
#     ecg = flatten_filter(df.ECG)
#     peak_intv = Rpeak_intervals(ecg)
#     fft(ecg)
#     non_af_dfs.append(peak_intv)


# df = pd.read_csv('mimic_perform_non_af_csv/mimic_perform_non_af_010_data.csv')

# # plt.plot(df.Time, df.ECG)
# # plt.title('Normal ECG segment')
# # plt.ylabel('Amplitude (mV)')
# # plt.xlabel('Time [s]')
# # plt.show()
# # peaks,_ = scipy.signal.find_peaks(df.PPG)
# # plt.plot(df.Time, df.PPG)
# # plt.plot(df.Time[peaks], df.PPG[peaks], "x")
# # plt.show()

# flat_filt_ecg = flatten_filter(df.ECG)

# Rpeak_intervals(flat_filt_ecg)

# fft(flat_filt_ecg)