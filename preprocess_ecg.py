

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
    # ecg = lowpass(ecg, max, sample_rate)
    # ecg = highpass(ecg, min, sample_rate)
    # only use bandpass for filtering
    ecg = bandpass(ecg, [min, max], sample_rate)
    return ecg

def interpolate(arr):
    nan_indices = np.where(np.isnan(arr))[0]
    
    for idx in nan_indices:
        left_idx = idx - 1
        right_idx = idx + 1

        # find closest non-nans
        while left_idx >= 0 and np.isnan(arr[left_idx]):
            left_idx -= 1
        while right_idx < len(arr) and np.isnan(arr[right_idx]):
            right_idx += 1

        if left_idx < 0:
            arr[0] = arr[right_idx]
        if right_idx >= len(arr):
            arr[len(arr)-1] = arr[left_idx]

        # interpolation
        if left_idx >= 0 and right_idx < len(arr):
            left_val = arr[left_idx]
            right_val = arr[right_idx]
            slope = (right_val - left_val) / (right_idx - left_idx)
            for i in range(left_idx + 1, right_idx):
                arr[i] = left_val + slope * (i - left_idx)

    return arr

def interp_flat(df, start, min_freq, max_freq):
    result = interpolate(df.ECG)
    result = flatten_filter(result, min_freq, max_freq)
    
    # fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
    # ax1.plot(df.Time, df.ECG)
    # ax1.set_title("Original Signal")
    # ax1.margins(0, .1)
    # ax1.grid(alpha=.5, ls='--')
    # ax2.plot(df.Time, result)
    # ax2.set_title("After Signal")
    # ax1.margins(0, .1)
    # ax2.grid(alpha=.5, ls='--')
    # plt.show()
    return result

# find intervals
def Rpeak_intervals(ecg, time):
    d_ecg, peaks_d_ecg = lab_funcs.decg_peaks(ecg, time)
    filt_peaks = lab_funcs.d_ecg_peaks(d_ecg, peaks_d_ecg, time, 0.5, 0.5)
    Rpeaks = lab_funcs.Rwave_peaks(ecg, d_ecg, filt_peaks, time)
    Rwave_t_peaks = lab_funcs.Rwave_t_peaks(time, Rpeaks)
    Rpeak_intervals = np.diff(Rwave_t_peaks)

    # fig, ((ax2, ax3, ax4)) = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
    # ax2.plot(time[0:len(time)-1], d_ecg, color = 'red')
    # ax2.plot(time[peaks_d_ecg], d_ecg[peaks_d_ecg], "x", color = 'g')
    # ax2.set_xlabel('Time [s]')
    # ax2.set_ylabel('Derivative of activation []')
    # ax2.set_title('R-wave peaks step 1: peaks of derivative of ECG')
    # ax2.grid(alpha=.1, ls='--')
    # ax3.plot(time[0:len(time)-1], d_ecg, color = 'red') 
    # ax3.plot(time[filt_peaks], d_ecg[filt_peaks], "x", color = 'g')
    # ax3.set_title('R-wave peaks step 2: d_ECG peaks')
    # ax3.set_ylabel('Derivative of activation []')
    # ax3.set_xlabel('Time [s]')
    # ax3.grid(alpha=.1, ls='--')
    # ax4.plot(time[0:len(time)-1], d_ecg, color = 'r', label = 'Derivative of ECG')
    # ax4.plot(time[filt_peaks], d_ecg[filt_peaks], "x", color = 'g')
    # ax4.set_ylabel('Activation Derivative []')
    # ax4.set_xlabel('Time [s]') 
    # ax4.set_title('R-wave peaks step 3: R-wave peaks')
    # ax4.plot(time, ecg, color = 'b', label = 'ECG')
    # ax4.plot(time[Rpeaks], ecg[Rpeaks], "x", color = 'y')
    # ax4.set_ylabel('Activation []')
    # ax4.grid(alpha=.1, ls='--')
    # plt.tight_layout()
    # plt.show()

    # avg_intv = np.mean(Rpeak_intervals)
    # min_intv = np.min(Rpeak_intervals)
    # max_intv = np.max(Rpeak_intervals)
    # print(avg_intv, min_intv, max_intv, len(Rwave_t_peaks))
    
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
    




# df = pd.read_csv('mimic_perform_non_af_csv/mimic_perform_non_af_001_data.csv')
# print(df.shape)

# plt.plot(df.Time, df.ECG)
# plt.title('Normal ECG segment')
# plt.ylabel('Amplitude (mV)')
# plt.xlabel('Time [s]')
# plt.show()
# peaks,_ = scipy.signal.find_peaks(df.PPG)
# plt.plot(df.Time, df.PPG)
# plt.plot(df.Time[peaks], df.PPG[peaks], "x")
# plt.show()

# min_freq = 5
# max_freq = 40
# n = 2000
# orig = df[:n]
# print(orig.shape)

# flat_filt_ecg = interp_flat(orig, 0, min_freq, max_freq)

# Rpeak_intervals(flat_filt_ecg, orig.Time)

# fft(flat_filt_ecg)