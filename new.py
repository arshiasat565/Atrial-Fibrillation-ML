import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs
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

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = sps.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_lp_data = sps.sosfiltfilt(sos, data)
    return filtered_lp_data

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_hp_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_hp_data

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_bp_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_bp_data

def interpolate(arr, start):
    nan_indices = np.where(np.isnan(arr))[0]
    # Interpolate NaN values
    for idx in nan_indices:
        left_idx = idx - 1
        right_idx = idx + 1

        # Find the closest non-NaN values
        while np.isnan(arr[start+left_idx]) and left_idx >= 0:
            left_idx -= 1
        while np.isnan(arr[start+right_idx]) and right_idx < start+len(arr):
            right_idx += 1

        # Perform interpolation
        if left_idx >= 0 and right_idx < start+len(arr):
            left_val = arr[start+left_idx]
            right_val = arr[start+right_idx]
            slope = (right_val - left_val) / (right_idx - left_idx)
            for i in range(left_idx + 1, right_idx):
                arr[start+i] = left_val + slope * (i - left_idx)

    return arr

af_dfs = []
non_af_dfs = []

df = pd.read_csv('mimic_perform_af_csv/mimic_perform_af_005_data.csv')

start = 0
ecg = df.ECG[start:1000]
ecg = interpolate(ecg, start)
# plt.plot(ecg)

# Apply a 40 Hz low-pass filter to the original data
filtered_lp = lowpass(ecg, 40, sample_rate)
# Apply 5 Hz high-pass filter
filtered_hp = highpass(filtered_lp, 5, sample_rate)
# Apply 5-40 Hz band-pass filter
filtered_bp = bandpass(filtered_hp, [5, 40], sample_rate)

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

d_ecg, peaks_d_ecg = lab_funcs.decg_peaks(filtered_bp, df.Time)
filt_peaks = lab_funcs.d_ecg_peaks(d_ecg, peaks_d_ecg, df.Time, 0.5, 0.5)
Rpeaks = lab_funcs.Rwave_peaks(filtered_bp, d_ecg, filt_peaks, df.Time)
Rpeak_intervals = np.diff(Rpeaks)
avg_intv = np.mean(Rpeak_intervals)
print(avg_intv)