

import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs_ecg

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

def flatten_filter(ecg, min, max, sample_rate=sample_rate):
    # ecg = lowpass(ecg, max, sample_rate)
    # ecg = highpass(ecg, min, sample_rate)
    # only use bandpass for filtering
    result = bandpass(ecg, [min, max], sample_rate)
    # fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
    # ax1.plot(ecg)
    # ax1.set_title("Original Signal")
    # ax1.margins(0, .1)
    # ax1.grid(alpha=.5, ls='--')
    # ax2.plot(result)
    # ax2.set_title("After Signal")
    # ax1.margins(0, .1)
    # ax2.grid(alpha=.5, ls='--')
    # plt.show()
    return result

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

# find intervals using lab_funcs
def Rpeak_intervals(ecg, time):
    d_ecg, peaks_d_ecg = lab_funcs_ecg.decg_peaks(ecg, time)
    filt_peaks, threshold = lab_funcs_ecg.d_ecg_peaks(d_ecg, peaks_d_ecg, 2, time, 0.5, 0.5)
    Rpeaks = lab_funcs_ecg.Rwave_peaks(ecg, d_ecg, filt_peaks, time)
    Rwave_t_peaks = lab_funcs_ecg.Rwave_t_peaks(time, Rpeaks)
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
    # ax3.axhline(threshold, color = 'black', label = 'threshold')
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

def data_init(min_freq, max_freq, length, start = 0):

    af_csv = glob.glob('mimic_perform_af_csv/*.csv')
    non_af_csv = glob.glob('mimic_perform_non_af_csv/*.csv')

    column_names = [
        "Time",
        "PPG",
        "ECG",
        "resp"
    ]

    ecgs = []
    Rpeak_intvs = []
    labels = []

    # interpolate and bandpass ecgs
    for csv in af_csv:
        df = pd.read_csv(csv)
        ecg = interp_flat(df, start, min_freq, max_freq)
        ecgs.append(ecg)
        labels.append(True)
        Rpeak_intvs.append(Rpeak_intervals(ecg, df.Time))

    for csv in non_af_csv:
        df = pd.read_csv(csv)
        ecg = interp_flat(df, start, min_freq, max_freq)
        ecgs.append(ecg)
        labels.append(False)
        Rpeak_intvs.append(Rpeak_intervals(ecg, df.Time))
        
    return ecgs, Rpeak_intvs, labels


def time_dependent_frequency(signal, sampling_rate, window_size=128, overlap=0.5):
    """
    Estimate the time-dependent frequency of a signal as the first moment of the power spectrogram
    using Short-Time Fourier Transform (STFT).
    
    Parameters:
    - signal: 1D numpy array representing the input signal.
    - sampling_rate: Sampling rate of the input signal.
    - window_size: Size of the window for STFT (default is 256).
    - overlap: Overlap between consecutive windows (0 to 1, default is 0.5).
    
    Generates:
    - frequencies: 1D numpy array representing the frequency axis.
    - times: 1D numpy array representing the time axis.
    - spectrogram: 2D numpy array representing the time-frequency spectrogram.
    
    Returns:
    - time_dep_freq: 1D numpy array representing the time-dependent frequency
    """
    freqs, times, spectrogram = sps.spectrogram(signal, fs=sampling_rate, window='hann',
                                                  nperseg=window_size, noverlap=int(window_size * overlap))
    
    # print(freqs.shape, times.shape, spectrogram.shape)
    # X, Y = np.meshgrid(times, freqs)
    # plt.pcolormesh(X, Y, 10 * np.log10(spectrogram), shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.colorbar(label='Power/Frequency (dB/Hz)')
    # plt.show()

    spectrogram_abs = np.abs(spectrogram)
    time_dep_freq = [np.sum(freqs * time_segm) / np.sum(time_segm) for time_segm in spectrogram_abs.T]
    # plt.plot(times, time_dep_freq)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Time-Dependent Frequency')
    # plt.grid(True)
    # plt.show()
    return time_dep_freq

def spectral_entropy(signal, sampling_rate, window_size=128, overlap=0.5):
    """
    Calculate spectral entropy of a signal.

    Parameters:
    - signal: 1D numpy array representing the input signal.
    - sampling_rate: Sampling rate of the input signal.
    - window_size: Size of the window for STFT (default is 256).
    - overlap: Overlap between consecutive windows (0 to 1, default is 0.5).
    
    Returns:
    - spectral_entropy values.
    """
    freqs, times, spectrogram = sps.spectrogram(signal, fs=sampling_rate, window='hann',
                                                  nperseg=window_size, noverlap=int(window_size * overlap))
    ps = np.abs(spectrogram)**2 # power spectrogram
    norm_ps = ps / np.sum(ps)
    spectral_entropy = -np.sum(norm_ps * np.log2(norm_ps), axis=0)
    return spectral_entropy