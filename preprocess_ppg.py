
import scipy.io
import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs_ppg
from sklearn.model_selection import train_test_split
import keras.metrics as km
from keras.models import Sequential
from keras.layers import Input


def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sps.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_bp_data = sps.sosfiltfilt(sos, data)
    return filtered_bp_data

def flatten_filter(ppg, min, max, sample_rate):
    # only use bandpass for filtering
    result = bandpass(ppg, [min, max], sample_rate)
    # fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
    # ax1.plot(ppg)
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
    ppg_segments = []
    time_segments = []
    result = interpolate(df.PPG)
    #split into 30s segments (ignore signal at 0)
    for i in range(1, len(result), length):
        ppg_segment = result[i:i+length]
        time_segment = df.Time[i:i+length]
        flatten_ppg = flatten_filter(ppg_segment, min_freq, max_freq, sample_rate)
        ppg_segments.append(flatten_ppg)
        time_segments.append(time_segment)
    
        # fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
        # ax1.plot(time_segment, ppg_segment)
        # ax1.set_title("Original Signal")
        # ax1.margins(0, .1)
        # ax1.grid(alpha=.5, ls='--')
        # ax2.plot(time_segment, flatten_ppg)
        # ax2.set_title("After Signal")
        # ax1.margins(0, .1)
        # ax2.grid(alpha=.5, ls='--')
        # plt.show()

    return ppg_segments, time_segments

# find intervals
def Rpeak_intervals(ppgs, times):
    #TODO: Use PPG specific
    Rpeak_intervals = []
    for ppg, time in zip(ppgs, times):
        time = time.values
        d_ppg, peaks_d_ppg = lab_funcs_ppg.dppg_peaks(ppg, time)
        filt_peaks, threshold = lab_funcs_ppg.d_ppg_peaks(d_ppg, peaks_d_ppg, 0, time, 0.5, 0.5)
        Rpeaks = lab_funcs_ppg.Rwave_peaks(ppg, d_ppg, filt_peaks, time)
        # Rwave_t_peaks = lab_funcs.Rwave_t_peaks(time, Rpeaks)
        Rpeak_diff = np.diff(Rpeaks)
        Rpeak_intervals = np.concatenate((Rpeak_intervals, Rpeak_diff))


        # fig, ((ax2, ax3, ax4)) = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
        # fig, ((ax3)) = plt.subplots(1, 1, figsize=(10, 3), sharex=True, sharey=True)
        # ax2.plot(time[0:len(time)-1], d_ppg, color = 'red')
        # ax2.plot(time[peaks_d_ppg], d_ppg[peaks_d_ppg], "x", color = 'g')
        # ax2.set_xlabel('Time [s]')
        # ax2.set_ylabel('Derivative of activation []')
        # ax2.set_title('R-wave peaks step 1: peaks of derivative of PPG')
        # ax2.grid(alpha=.1, ls='--')
        # ax3.plot(time[0:len(time)-1], d_ppg, color = 'red') 
        # ax3.plot(time[filt_peaks], d_ppg[filt_peaks], "x", color = 'g')
        # ax3.axhline(threshold, color = 'black', label = 'threshold')
        # ax3.set_title('R-wave peaks step 2: d_PPG peaks')
        # ax3.set_ylabel('Derivative of activation []')
        # ax3.set_xlabel('Time [s]')
        # ax3.grid(alpha=.1, ls='--')
        # ax4.plot(time[0:len(time)-1], d_ppg, color = 'r', label = 'Derivative of PPG')
        # ax4.plot(time[filt_peaks], d_ppg[filt_peaks], "x", color = 'g')
        # ax4.set_ylabel('Activation Derivative []')
        # ax4.set_xlabel('Time [s]') 
        # ax4.set_title('R-wave peaks step 3: R-wave peaks')
        # ax4.plot(time, ppg, color = 'b', label = 'PPG')
        # ax4.plot(time[Rpeaks], ppg[Rpeaks], "x", color = 'y')
        # ax4.set_ylabel('Activation []')
        # ax4.grid(alpha=.1, ls='--')
        # plt.tight_layout()
        # plt.show()
    
    return Rpeak_intervals

def fft(ppg, sample_rate):
    # Compute the FFT
    fft_result = np.fft.fft(ppg)
    frequencies = np.fft.fftfreq(len(fft_result), 1/sample_rate)  # Assuming a sampling frequency of 1000 Hz

    # Plot the magnitude spectrum
    # plt.figure(figsize=(10, 6))
    # plt.plot(frequencies, np.abs(fft_result))
    # plt.title('FFT of Filtered PPG Signal')
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

    ppgs = []
    times = []
    Rpeak_intvs = []
    segment_labels = []
    interval_labels = []

    print("database")
    # interpolate and bandpass ppgs
    for csv in af_csv:
        # print(csv)
        df = pd.read_csv(csv)
        ppg_segments, time_segments = interp_flat(df, length, min_freq, max_freq, sample_rate)
        for ppg_segment, time_segment in zip(ppg_segments, time_segments):
            ppgs.append(ppg_segment)
            times.append(time_segment)
            segment_labels.append(True)
        Rpeak_intvs.append(Rpeak_intervals(ppg_segments, time_segments))
        interval_labels.append(True)

    for csv in non_af_csv:
        # print(csv)
        df = pd.read_csv(csv)
        ppg_segments, time_segments = interp_flat(df, length, min_freq, max_freq, sample_rate)
        for ppg_segment, time_segment in zip(ppg_segments, time_segments):
            ppgs.append(ppg_segment)
            times.append(time_segment)
            segment_labels.append(False)
        Rpeak_intvs.append(Rpeak_intervals(ppg_segments, time_segments))
        interval_labels.append(False)

    return ppgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate


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

# data from generator (2 pulses); 100 RR intervals (80s) cut to signal_length, AF/nonAF data 1024 each 
def large_data(signal_length, size=None):
    sample_rate = 250
    af_models = glob.glob('model/1/*.mat')
    non_af_models = glob.glob('model/0/*.mat')
    time = pd.Series(np.linspace(0, signal_length / sample_rate, signal_length, endpoint=False))

    signal_labels = []
    parameters = []
    ppgs = []
    labels = []
    Rpeak_intvs = []

    print("generated 1")
    # data init
    for model in af_models:
        data = scipy.io.loadmat(model)
        signal_labels.append(data['labels'])
        parameters.append(data['parameters'])
        signals = data['signals'][0, 0]

        ppg = np.array(signals['PPG'])[:, 0:signal_length] # ['PPG'] 1&2
        ppg1 = flatten_filter(ppg[0], 1, 40, sample_rate=sample_rate)
        ppg2 = flatten_filter(ppg[1], 1, 40, sample_rate=sample_rate)
        ppg = np.stack((ppg1, ppg2))
        ppgs.append(ppg)
        Rpeak_intv1 = Rpeak_intervals([ppg1], [time])
        Rpeak_intv2 = Rpeak_intervals([ppg2], [time])
        max_length = max([len(Rpeak_intv1), len(Rpeak_intv2)])
        Rpeak_intv1 = np.pad(Rpeak_intv1, (0, max_length - len(Rpeak_intv1)), constant_values=0)
        Rpeak_intv2 = np.pad(Rpeak_intv2, (0, max_length - len(Rpeak_intv2)), constant_values=0)
        Rpeak_intvs.append(np.stack((Rpeak_intv1, Rpeak_intv2)))
        labels.append(True)
        
    if size != None:
        ppgs = ppgs[:size//2]
        labels = labels[:size//2]

    for model in non_af_models:
        data = scipy.io.loadmat(model)
        signal_labels.append(data['labels'])
        parameters.append(data['parameters'])
        signals = data['signals'][0, 0]

        ppg = np.array(signals['PPG'])[:, 0:signal_length] # ['PPG'] 1&2
        ppg1 = flatten_filter(ppg[0], 1, 40, sample_rate=sample_rate)
        ppg2 = flatten_filter(ppg[1], 1, 40, sample_rate=sample_rate)
        ppg = np.stack((ppg1, ppg2))
        ppgs.append(ppg)
        Rpeak_intv1 = Rpeak_intervals([ppg1], [time])
        Rpeak_intv2 = Rpeak_intervals([ppg2], [time])
        max_length = max([len(Rpeak_intv1), len(Rpeak_intv2)])
        Rpeak_intv1 = np.pad(Rpeak_intv1, (0, max_length - len(Rpeak_intv1)), constant_values=0)
        Rpeak_intv2 = np.pad(Rpeak_intv2, (0, max_length - len(Rpeak_intv2)), constant_values=0)
        Rpeak_intvs.append(np.stack((Rpeak_intv1, Rpeak_intv2)))
        labels.append(False)
        
    if size != None:
        ppgs = ppgs[:size]
        labels = labels[:size]

    # print(len(ppgs))

    ppgs, labels = np.array(ppgs), np.array(labels)
    # labels = labels.reshape(len(labels), 1)
    return ppgs, labels, Rpeak_intvs, sample_rate

# for patient database
def feature_extraction_db(ppgs, sample_rate):
    # feature extraction
    ffts = np.array([fft(ppg, sample_rate) for ppg in ppgs])
    # ffts = ffts.swapaxes(1, 2)
    print(ffts.shape)

    # ppg instantaneous frequencies (time-dependent)
    infs = np.array([time_dependent_frequency(ppg, sample_rate) for ppg in ppgs])
    inf_mean = np.mean(infs)
    inf_std = np.std(infs)
    infs = np.array([(x - inf_mean) / inf_std for x in infs])

    # ppg spectral entropies
    ses = np.array([spectral_entropy(ppg, sample_rate) for ppg in ppgs])
    se_mean = np.mean(ses)
    se_std = np.std(ses)
    ses = np.array([(x - se_mean) / se_std for x in ses])

    print(infs.shape, ses.shape)

    return ffts, infs, ses

# for computer generated data
def feature_extraction_gen(ppgs, labels, sample_rate):
    # feature extraction
    fft1s = np.array([fft(ppg[0], sample_rate) for ppg in ppgs])
    fft2s = np.array([fft(ppg[1], sample_rate) for ppg in ppgs])
    ffts = np.concatenate((fft1s, fft2s))
    print(ffts.shape)

    # ppg instantaneous frequencies (time-dependent)
    inf1s = np.array([time_dependent_frequency(ppg[0], sample_rate) for ppg in ppgs])
    inf_mean = np.mean(inf1s)
    inf_std = np.std(inf1s)
    inf1s = np.array([(x - inf_mean) / inf_std for x in inf1s])
    inf2s = np.array([time_dependent_frequency(ppg[1], sample_rate) for ppg in ppgs])
    inf_mean = np.mean(inf2s)
    inf_std = np.std(inf2s)
    inf2s = np.array([(x - inf_mean) / inf_std for x in inf2s])

    # ppg spectral entropies
    se1s = np.array([spectral_entropy(ppg[0], sample_rate) for ppg in ppgs])
    se_mean = np.mean(se1s)
    se_std = np.std(se1s)
    se1s = np.array([(x - se_mean) / se_std for x in se1s])
    se2s = np.array([spectral_entropy(ppg[1], sample_rate) for ppg in ppgs])
    se_mean = np.mean(se2s)
    se_std = np.std(se2s)
    se2s = np.array([(x - se_mean) / se_std for x in se2s])

    print(inf1s.shape, inf2s.shape, se1s.shape, se2s.shape)
    infs = np.concatenate((inf1s, inf2s))
    ses = np.concatenate((se1s, se2s))
    labels = np.concatenate((labels, labels))

    return ffts, infs, ses, labels

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

def split_dataset(features, labels):
    features, labels = np.array(features), np.array(labels)
    print(features.shape, labels.shape)
    feature_train, feature_test, feature_label_train, feature_label_test = train_test_split(features, labels, test_size=0.2)
    feature_train, feature_val, feature_label_train, feature_label_val = train_test_split(feature_train, feature_label_train, test_size=0.2)
    print((feature_train.shape), (feature_val.shape), (feature_test.shape))
    print((feature_label_train.shape), (feature_label_val.shape), (feature_label_test.shape))

    if features.ndim > 2:
        num = features.shape[2]
    else:
        num = 1
    model = Sequential()
    model.add(Input(shape=(features.shape[1], num)))

    return model, (feature_train, feature_val, feature_test, feature_label_train, feature_label_val, feature_label_test)

def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

metrics = [
    'accuracy', 'precision', 'recall', km.AUC(curve='ROC')
]

def model_fit(model, feature_labels):
    feature_train, feature_val, feature_test, feature_label_train, feature_label_val, feature_label_test = feature_labels

    accs = []
    losses = []
    results = []
    for i in range(10):
        model.fit(feature_train, feature_label_train, validation_data=(feature_val, feature_label_val), epochs=100, verbose=0)

        result = model.evaluate(feature_test, feature_label_test, verbose=1)
        result.append(f1_score(result[2], result[3]))
        # print(result)
        results.append(result)

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