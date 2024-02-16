import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs
import basic
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
min_freq = 5
max_freq = 40
start = 0
length = 150000 # 20 mins

# filters
for csv in af_csv:
    df = pd.read_csv(csv)
    ecg = basic.interpolate(df.ECG, 0)
    ecg = basic.flatten_filter(ecg, min_freq, max_freq)
    ecgs.append(ecg)
    labels.append(True)

print('non')
for csv in non_af_csv:
    df = pd.read_csv(csv)
    ecg = basic.interpolate(df.ECG, 0)
    ecg = basic.flatten_filter(ecg, min_freq, max_freq)
    ecgs.append(ecg)
    labels.append(False)

# Rpeaks
for ecg in ecgs:
    d_ecg, peaks_d_ecg = lab_funcs.decg_peaks(ecg, df.Time)
    filt_peaks = lab_funcs.d_ecg_peaks(d_ecg, peaks_d_ecg, df.Time, 0.5, 0.5)
    ecg_Rpeaks = lab_funcs.Rwave_peaks(ecg, d_ecg, filt_peaks, df.Time)
    Rwave_t_peaks = lab_funcs.Rwave_t_peaks(df.Time, ecg_Rpeaks)
    Rpeak_intervals = np.diff(Rwave_t_peaks)
    avg_intv = np.mean(Rpeak_intervals)
    Rpeak_intvs.append(Rpeak_intervals)

print(len(ecgs), len(Rpeak_intvs), len(labels))

ecg_train, ecg_test, label_train, label_test = train_test_split(Rpeak_intvs, labels, test_size=0.2, random_state=8964)

clas = svm.SVC(kernel="linear")

print(len(ecg_train), len(label_train))

clas.fit(ecg_train, label_train)

label_pred = clas.predict(ecg_test)

acc = accuracy_score(label_test, label_pred)
print(acc)