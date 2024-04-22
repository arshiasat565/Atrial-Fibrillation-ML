import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs_ecg
import preprocess_ecg
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}
show_training = False

min_freq = 5
max_freq = 40
start = 0
length = 150000 # 20 mins

ecgs, Rpeak_intvs, labels = preprocess_ecg.data_init(min_freq, max_freq, length)

clas = svm.SVC(kernel="rbf", probability=True)

#10 fold cross validation svm, use full ecg (base)
scores = cross_val_score(clas, ecgs, labels, cv=10)

print("base Cross-Validation:")
scores = cross_validate(clas, ecgs, labels, cv=10, scoring=scoring, return_train_score=show_training)
for metric_name, score in list(scores.items())[2:]:
    # print(score)
    print(f"Mean {metric_name}: {score.mean()} (±{score.std()})")


#split ecg
sample_size = 3750 # 30 seconds
ecg_samples = []
sample_labels = []
sample_size_sec = sample_size / 125
print(f"\nBy {sample_size_sec}s ecg samples")

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
print("kf Cross-Validation:")
scores = cross_validate(clas, ecg_samples, sample_labels, cv=kf, scoring=scoring, return_train_score=show_training)
for metric_name, score in list(scores.items())[2:]:
    # print(score)
    print(f"Mean {metric_name}: {score.mean()} (±{score.std()})")

# shuffle
shuffle_split = ShuffleSplit(n_splits=10)
print("shuffle Cross-Validation:")
scores = cross_validate(clas, ecg_samples, sample_labels, cv=shuffle_split, scoring=scoring, return_train_score=show_training)
for metric_name, score in list(scores.items())[2:]:
    # print(score)
    print(f"Mean {metric_name}: {score.mean()} (±{score.std()})")


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
print("kf Cross-Validation:")
scores = cross_validate(clas, intv_samples, sample_labels, cv=kf, scoring=scoring, return_train_score=show_training)
for metric_name, score in list(scores.items())[2:]:
    # print(score)
    print(f"Mean {metric_name}: {score.mean()} (±{score.std()})")

# shuffle
shuffle_split = ShuffleSplit(n_splits=10)
print("shuffle Cross-Validation:")
scores = cross_validate(clas, intv_samples, sample_labels, cv=shuffle_split, scoring=scoring, return_train_score=show_training)
for metric_name, score in list(scores.items())[2:]:
    # print(score)
    print(f"Mean {metric_name}: {score.mean()} (±{score.std()})")
