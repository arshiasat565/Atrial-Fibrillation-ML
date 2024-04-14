import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs
import preprocess_ppg
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
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

# interpolate and bandpass ecgs
for csv in af_csv:
    df = pd.read_csv(csv)
    ecg = preprocess_ppg.interp_flat(df, start, min_freq, max_freq)
    ecgs.append(ecg)
    labels.append(True)
    Rpeak_intervals = preprocess_ppg.Rpeak_intervals(ecg, df.Time)
    Rpeak_intvs.append(Rpeak_intervals)

for csv in non_af_csv:
    print(csv)
    df = pd.read_csv(csv)
    ecg = preprocess_ppg.interp_flat(df, start, min_freq, max_freq)
    ecgs.append(ecg)
    labels.append(False)
    Rpeak_intervals = preprocess_ppg.Rpeak_intervals(ecg, df.Time)
    Rpeak_intvs.append(Rpeak_intervals)

# Rpeak_intervals
# for ecg in ecgs:
#     Rpeak_intervals = preprocess.Rpeak_intervals(ecg, df.Time)
#     Rpeak_intvs.append(Rpeak_intervals)

# for i, Rpeak_intv in enumerate(Rpeak_intvs):
#     print(i, len(Rpeak_intv))

clas = svm.SVC(kernel="rbf")

#10 fold cross validation svm, use full ecg (base)
scores = cross_val_score(clas, ecgs, labels, cv=10)

print("base Cross-Validation Scores:", scores)

mean_accuracy = np.mean(scores)
print("Mean Accuracy:", mean_accuracy)


#split ecg
print("\nBy ecg samples")
sample_size = 3125 # 20 seconds
ecg_samples = []
sample_labels = []

for i, ecg in enumerate(ecgs):
    for j in range(0, len(ecg)-1, sample_size): # last signal removed
        ecg_sample = ecg[j:j+sample_size]
        # print(len(ecg_sample))
        ecg_samples.append(ecg_sample)
        sample_labels.append(labels[i])
print("sample count:", len(ecg_samples))

# 10 fold cross validation svm, use ecg samples
# kfold
kf = KFold(n_splits=10, shuffle=True)

scores = cross_val_score(clas, ecg_samples, sample_labels, cv=kf)

print("kf Cross-Validation Scores:", scores)

mean_accuracy = np.mean(scores)
print("Mean Accuracy:", mean_accuracy)

# shuffle
shuffle_split = ShuffleSplit(n_splits=10)
scores = cross_val_score(clas, ecg_samples, sample_labels, cv=shuffle_split)

print("shuffle Cross-Validation Scores:", scores)

mean_accuracy = np.mean(scores)
print("Mean Accuracy:", mean_accuracy)


#split Rpeak_intvs
print("\nBy Rpeak_intv samples")
sample_size = 25
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

# 10 fold cross validation svm, use Rpeak_intv samples
# kfold
kf = KFold(n_splits=10, shuffle=True)

scores = cross_val_score(clas, intv_samples, sample_labels, cv=kf)

print("kf Cross-Validation Scores:", scores)

mean_accuracy = np.mean(scores)
print("Mean Accuracy:", mean_accuracy)

# shuffle
shuffle_split = ShuffleSplit(n_splits=10)
scores = cross_val_score(clas, intv_samples, sample_labels, cv=shuffle_split)

print("shuffle Cross-Validation Scores:", scores)

mean_accuracy = np.mean(scores)
print("Mean Accuracy:", mean_accuracy)