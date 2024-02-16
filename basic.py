

import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs

sample_rate = 120

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = sps.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_lp_data = sps.sosfiltfilt(sos, data)
    return filtered_lp_data

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = sps.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_hp_data = sps.sosfiltfilt(sos, data)
    return filtered_hp_data

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sps.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_bp_data = sps.sosfiltfilt(sos, data)
    return filtered_bp_data

def flatten_filter(ecg, min, max):
    ecg = lowpass(ecg, max, sample_rate)
    ecg = highpass(ecg, min, sample_rate)
    ecg = bandpass(ecg, [min, max], sample_rate)
    return ecg

def interpolate(arr, start):
    nan_indices = np.where(np.isnan(arr))[0]
    
    for idx in nan_indices:
        left_idx = idx - 1
        right_idx = idx + 1

        # find closest non-nans
        while np.isnan(arr[start+left_idx]) and left_idx >= 0:
            left_idx -= 1
        while np.isnan(arr[start+right_idx]) and right_idx < start+len(arr):
            right_idx += 1

        # interpolation
        if left_idx >= 0 and right_idx < start+len(arr):
            left_val = arr[start+left_idx]
            right_val = arr[start+right_idx]
            slope = (right_val - left_val) / (right_idx - left_idx)
            for i in range(left_idx + 1, right_idx):
                arr[start+i] = left_val + slope * (i - left_idx)

    return arr

# find intervals
def Rpeak_intervals(ecg, time):
    d_ecg, peaks_d_ecg = lab_funcs.decg_peaks(ecg, time)
    filt_peaks = lab_funcs.d_ecg_peaks(d_ecg, peaks_d_ecg, time, 0.5, 0.5)
    Rpeaks = lab_funcs.Rwave_peaks(ecg, d_ecg, filt_peaks, time)
    Rwave_t_peaks = lab_funcs.Rwave_t_peaks(time, Rpeaks)
    Rpeak_intervals = np.diff(Rwave_t_peaks)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 3), sharex=True, sharey=True)
    # ax1.plot(ecg)
    # ax1.set_title("Original Signal")
    # ax1.margins(0, .1)
    # ax1.grid(alpha=.5, ls='--')
    ax2.plot(time[0:len(time)-1], d_ecg, color = 'red')
    ax2.plot(time[peaks_d_ecg], d_ecg[peaks_d_ecg], "x", color = 'g')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Derivative of activation []')
    ax2.set_title('R-wave peaks step 1: peaks of derivative of ECG')
    ax2.grid(alpha=.1, ls='--')
    ax3.plot(time[0:len(time)-1], d_ecg, color = 'red') 
    ax3.plot(time[filt_peaks], d_ecg[filt_peaks], "x", color = 'g')
    #plt.axhline(meanpeaks_d_ecg, color = 'b')
    #plt.axhline(max_d_ecg, color = 'b')
    # thres = ax3.axhline(threshold, color = 'black', label = 'threshold')
    ax3.set_title('R-wave peaks step 2: d_ECG peaks')
    ax3.set_ylabel('Derivative of activation []')
    ax3.set_xlabel('Time [s]')
    ax3.grid(alpha=.1, ls='--')
    ax4.plot(time[0:len(time)-1], d_ecg, color = 'r', label = 'Derivative of ECG')
    ax4.plot(time[filt_peaks], d_ecg[filt_peaks], "x", color = 'g')
    ax4.set_ylabel('Activation Derivative []')
    ax4.set_xlabel('Time [s]') 
    ax4.set_title('R-wave peaks step 3: R-wave peaks')
    ax4.plot(time, ecg, color = 'b', label = 'ECG')
    ax4.plot(time[Rpeaks], ecg[Rpeaks], "x", color = 'y')
    ax4.set_ylabel('Activation []')
    ax4.grid(alpha=.1, ls='--')
    plt.tight_layout()
    plt.show()

    avg_intv = np.mean(Rpeak_intervals)
    min_intv = np.min(Rpeak_intervals)
    max_intv = np.max(Rpeak_intervals)
    print(avg_intv, min_intv, max_intv)
    
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