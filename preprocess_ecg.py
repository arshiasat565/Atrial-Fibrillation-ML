
import scipy.io
import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs_ecg
from sklearn.model_selection import train_test_split, ShuffleSplit
import keras.api.metrics as km
from keras.api.models import Sequential
from keras.api.layers import Input


def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sps.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_bp_data = sps.sosfiltfilt(sos, data)
    return filtered_bp_data

def flatten_filter(ecg, min, max, sample_rate):
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

def interp_flat(df, length, min_freq, max_freq, sample_rate):
    ecg_segments = []
    time_segments = []
    result = interpolate(df.ECG)
    #split into 30s segments (ignore signal at 0)
    for i in range(1, len(result), length):
        ecg_segment = result[i:i+length]
        time_segment = df.Time[i:i+length]
        flatten_ecg = flatten_filter(ecg_segment, min_freq, max_freq, sample_rate)
        ecg_segments.append(flatten_ecg)
        time_segments.append(time_segment)
    
        # fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
        # ax1.plot(time_segment, ecg_segment)
        # ax1.set_title("Original Signal")
        # ax1.margins(0, .1)
        # ax1.grid(alpha=.5, ls='--')
        # ax2.plot(time_segment, flatten_ecg)
        # ax2.set_title("After Signal")
        # ax1.margins(0, .1)
        # ax2.grid(alpha=.5, ls='--')
        # plt.show()

    return ecg_segments, time_segments

# find intervals using lab_funcs
def Rpeak_intervals(ecgs, times):
    Rpeak_intervals = []
    for ecg, time in zip(ecgs, times):
        time = time.values
        d_ecg, peaks_d_ecg = lab_funcs_ecg.decg_peaks(ecg, time)
        filt_peaks, threshold = lab_funcs_ecg.d_ecg_peaks(d_ecg, peaks_d_ecg, 2, time, 0.5, 0.5)
        Rpeaks = lab_funcs_ecg.Rwave_peaks(ecg, d_ecg, filt_peaks, time)
        # Rwave_t_peaks = lab_funcs.Rwave_t_peaks(time, Rpeaks)
        Rpeak_diff = np.diff(Rpeaks)
        Rpeak_intervals = np.concatenate((Rpeak_intervals, Rpeak_diff))


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
        # ax4.plot(time[Rpeaks], ecg[Rpeaks], "x", color = 'black')
        # ax4.set_ylabel('Activation []')
        # ax4.grid(alpha=.1, ls='--')
        # plt.tight_layout()
        # plt.show()
    
    return Rpeak_intervals

def fft(ecg, sample_rate):
    # Compute the FFT
    fft_result = np.fft.fft(ecg)
    frequencies = np.fft.fftfreq(len(fft_result), 1/sample_rate)  # Assuming a sampling frequency of 1000 Hz

    # Plot the magnitude spectrum
    # plt.figure(figsize=(10, 6))
    # plt.plot(frequencies, np.abs(fft_result))
    # plt.title('FFT of Filtered ECG Signal')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.show()
    fft_result = np.abs(fft_result)
    return fft_result

# data from MIMIC perform dataset; 20 mins 125Hz 19 AF, 16 nonAF patients
def data_init(min_freq, max_freq, length, start = 0):
    sample_rate = 125
    af_csv = glob.glob('mimic_perform_af_csv/*.csv')
    non_af_csv = glob.glob('mimic_perform_non_af_csv/*.csv')

    ecgs = []
    times = []
    Rpeak_intvs = []
    segment_labels = []
    interval_labels = []

    print("database")
    # interpolate and bandpass ecgs
    for csv in af_csv:
        df = pd.read_csv(csv)
        ecg_segments, time_segments = interp_flat(df, length, min_freq, max_freq, sample_rate)
        for ecg_segment, time_segment in zip(ecg_segments, time_segments):
            ecgs.append(ecg_segment)
            times.append(time_segment)
            segment_labels.append(True)
        Rpeak_intvs.append(Rpeak_intervals(ecg_segments, time_segments))
        interval_labels.append(True)

    for csv in non_af_csv:
        df = pd.read_csv(csv)
        ecg_segments, time_segments = interp_flat(df, length, min_freq, max_freq, sample_rate)
        for ecg_segment, time_segment in zip(ecg_segments, time_segments):
            ecgs.append(ecg_segment)
            times.append(time_segment)
            segment_labels.append(False)
        Rpeak_intvs.append(Rpeak_intervals(ecg_segments, time_segments))
        interval_labels.append(False)

    return ecgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate


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
    # plt.title('Instantaneous Frequency')
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
    # plt.plot(times, spectral_entropy)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Spectral Entrophy')
    # plt.title('Spectral Entropy (AF)')
    # plt.grid(True)
    # plt.show()
    return spectral_entropy

# data from generator; 100 RR intervals (80s) cut to signal_length, AF/nonAF data 1024 each 
def large_data(signal_length, size=None):
    sample_rate = 250
    af_models = glob.glob('model/1/*.mat')
    non_af_models = glob.glob('model/0/*.mat')
    time = pd.Series(np.linspace(0, signal_length / sample_rate, signal_length, endpoint=False))

    signal_labels = []
    parameters = []
    ecgs = []
    labels = []
    Rpeak_intvs = []

    print("generated 1")
    # data init
    for model in af_models:
        data = scipy.io.loadmat(model)
        signal_labels.append(data['labels'])
        parameters.append(data['parameters'])
        signals = data['signals'][0, 0]

        ecg_list = signals[1][0] # ['multileadECG'] I (may change when noise diff)
        ecg = [i[0] for i in ecg_list]
        # if len(ecg) < signal_length:
        #     print(len(ecg))
        ecg = np.array(ecg[0:signal_length])
        ecg = ecg.reshape(len(ecg))
        ecg = flatten_filter(ecg, 1, 40, sample_rate=sample_rate)
        ecgs.append(ecg)
        Rpeak_intvs.append(Rpeak_intervals([ecg], [time]))
        labels.append(True)

    if size != None:
        ecgs = ecgs[:size//2]
        labels = labels[:size//2]

    for model in non_af_models:
        data = scipy.io.loadmat(model)
        signal_labels.append(data['labels'])
        parameters.append(data['parameters'])
        signals = data['signals'][0, 0]

        ecg_list = signals[1][0] # ['multileadECG'] I (may change when noise diff)
        ecg = [i[0] for i in ecg_list]
        # if len(ecg) < signal_length:
        #     print(len(ecg))
        ecg = np.array(ecg[0:signal_length])
        ecg = ecg.reshape(len(ecg))
        ecg = flatten_filter(ecg, 1, 40, sample_rate=sample_rate)
        ecgs.append(ecg)
        Rpeak_intvs.append(Rpeak_intervals([ecg], [time]))
        labels.append(False)

    if size != None:
        ecgs = ecgs[:size]
        labels = labels[:size]

    # print(len(ecgs))

    labels = np.array(labels)
    # labels = labels.reshape(len(labels), 1)
    return ecgs, labels, Rpeak_intvs, sample_rate

def feature_extraction(ecgs, sample_rate):
    # feature extraction
    ffts = np.array([fft(ecg, sample_rate) for ecg in ecgs])
    print(ffts.shape)

    # ecg instantaneous frequencies (time-dependent)
    infs = np.array([time_dependent_frequency(ecg, sample_rate) for ecg in ecgs])
    inf_mean = np.mean(infs)
    inf_std = np.std(infs)
    infs = np.array([(x - inf_mean) / inf_std for x in infs])

    # ecg spectral entropies
    ses = np.array([spectral_entropy(ecg, sample_rate) for ecg in ecgs])
    se_mean = np.mean(ses)
    se_std = np.std(ses)
    ses = np.array([(x - se_mean) / se_std for x in ses])

    return ffts, infs, ses

def split_Rpeak_intvs(Rpeak_intvs, labels):
    # split Rpeak_intvs
    print("\nBy Rpeak_intv samples")
    sample_size = 10
    intv_samples = []
    sample_labels = []

    for i, Rpeak_intv in enumerate(Rpeak_intvs):
        for j in range(0, len(Rpeak_intv), sample_size):
            intv_sample = Rpeak_intv[j:j+sample_size]
            # print(len(ecg_sample))
            if len(intv_sample) == sample_size:
                intv_samples.append(intv_sample)
                sample_labels.append(labels[i])
    print("sample count:", len(intv_samples))

    return(intv_samples, sample_labels)

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

metrics = [
    'accuracy', 'precision', 'recall', km.AUC(curve='ROC')
]

def model_fit(base_model, callbacks, features, labels):

    features, labels = np.array(features), np.array(labels)
    print(features.shape, labels.shape)
    shuffle_split = ShuffleSplit(n_splits=10)

    accs = []
    losses = []
    results = []

    fold = 1
    for train_index, test_index in shuffle_split.split(features):
        print(f"Fold {fold}", train_index.size, test_index.size)
        fold += 1

        feature_train, feature_test = features[train_index], features[test_index]
        feature_label_train, feature_label_test = labels[train_index], labels[test_index]
    
        feature_train, feature_val, feature_label_train, feature_label_val = train_test_split(feature_train, feature_label_train, test_size=0.2)

        print((feature_train.shape), (feature_val.shape), (feature_test.shape))
        print((feature_label_train.shape), (feature_label_val.shape), (feature_label_test.shape))

        model = base_model(features)
        history = model.fit(feature_train, feature_label_train, validation_data=(feature_val, feature_label_val), epochs=100, verbose=0, callbacks=callbacks)
        result = model.evaluate(feature_test, feature_label_test, verbose=1)
        result.append(f1_score(result[2], result[3]))
        results.append(result)
        actual_epochs = len(history.history['loss'])
        print(f"Actual number of epochs run: {actual_epochs}")

    metric_names = np.concatenate((['loss'], metrics, ['f1_score']))
    np.set_printoptions(suppress=True)
    avg_results = np.average(results, axis=0)
    print("Average metrics:")
    for name, value in zip(metric_names, avg_results):
        print(f'{name}: {value:.4f}')

    #     loss, accuracy = model.evaluate(feature_test, feature_label_test)
    #     print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    #     losses.append(loss)
    #     accs.append(accuracy)
    # print(np.average(losses), np.average(accs))