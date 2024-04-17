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
    "PPG",
    "resp"
]

ppgs = []
times = []
Rpeak_intvs = []
segment_labels = []
interval_labels = []
min_freq = 0.5
max_freq = 5
start = 0
length = 3750 # 30 secs

# interpolate and bandpass ppgs
for csv in af_csv:
    print(csv)
    df = pd.read_csv(csv)
    ppg_segments, time_segments = preprocess_ppg.interp_flat(df, length, min_freq, max_freq)
    for ppg_segment, time_segment in zip(ppg_segments, time_segments):
        ppgs.append(ppg_segment)
        times.append(time_segment)
        segment_labels.append(True)
    Rpeak_intervals = preprocess_ppg.Rpeak_intervals(ppg_segments, time_segments)
    Rpeak_intvs.append(Rpeak_intervals)
    interval_labels.append(True)

for csv in non_af_csv:
    print(csv)
    df = pd.read_csv(csv)
    ppg_segments, time_segments = preprocess_ppg.interp_flat(df, length, min_freq, max_freq)
    for ppg_segment, time_segment in zip(ppg_segments, time_segments):
        ppgs.append(ppg_segment)
        times.append(time_segment)
        segment_labels.append(False)
    Rpeak_intervals = preprocess_ppg.Rpeak_intervals(ppg_segments, time_segments)
    Rpeak_intvs.append(Rpeak_intervals)
    interval_labels.append(False)

# Rpeak_intervals
# for ppg in ppgs:
#     Rpeak_intervals = preprocess.Rpeak_intervals(ppg, df.Time)
#     Rpeak_intvs.append(Rpeak_intervals)

# for i, Rpeak_intv in enumerate(Rpeak_intvs):
#     print(i, len(Rpeak_intv))

clas = svm.SVC(kernel="rbf")

# #10 fold cross validation svm, use full ppg (base)
# scores = cross_val_score(clas, ppgs, labels, cv=10)

# print("base Cross-Validation Scores:", scores)

# mean_accuracy = np.mean(scores)
# print("Mean Accuracy:", mean_accuracy)


# split ppg
print("\nBy ppg samples")
print("sample count:", len(ppgs))

# 10 fold cross validation svm, use ppg samples
# kfold
kf = KFold(n_splits=10, shuffle=True)

scores = cross_val_score(clas, ppgs, segment_labels, cv=kf)

print("kf Cross-Validation Scores:", scores)

mean_accuracy = np.mean(scores)
print("Mean Accuracy:", mean_accuracy)

# shuffle
shuffle_split = ShuffleSplit(n_splits=10)
scores = cross_val_score(clas, ppgs, segment_labels, cv=shuffle_split)

print("shuffle Cross-Validation Scores:", scores)

mean_accuracy = np.mean(scores)
print("Mean Accuracy:", mean_accuracy)


#split Rpeak_intvs
print("\nBy Rpeak_intv samples")
sample_size = 10
intv_samples = []
sample_labels = []

#TODO: Add Padding
for i, Rpeak_intv in enumerate(Rpeak_intvs):
    for j in range(0, len(Rpeak_intv), sample_size):
        intv_sample = Rpeak_intv[j:j+sample_size]
        if len(intv_sample) == sample_size:
            intv_samples.append(intv_sample)
            sample_labels.append(interval_labels[i])
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