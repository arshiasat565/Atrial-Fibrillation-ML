import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs_ppg
import preprocess_ppg
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
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

min_freq = 0.5
max_freq = 5
start = 0
length = 3750 # 30 secs

ppgs, times, Rpeak_intvs, segment_labels, interval_labels = preprocess_ppg.data_init(min_freq, max_freq, length)

clas = svm.SVC(kernel="rbf", probability=True)

# #10 fold cross validation svm, use full ppg (N/A, segmented by default)
# scores = cross_val_score(clas, ppgs, labels, cv=10)

# print("base Cross-Validation Scores:", scores)

# mean_accuracy = np.mean(scores)
# print("Mean Accuracy:", mean_accuracy)

length_sec = length / 125
# split ppg
print(f"\nBy {length_sec}s ppg samples")
print("sample count:", len(ppgs))

# 10 fold cross validation svm, use ppg samples
# kfold
kf = KFold(n_splits=10, shuffle=True)
print("kf Cross-Validation:")
scores = cross_validate(clas, ppgs, segment_labels, cv=kf, scoring=scoring, return_train_score=show_training)
for metric_name, score in list(scores.items())[2:]:
    # print(score)
    print(f"Mean {metric_name}: {score.mean()} (±{score.std()})")

# shuffle
shuffle_split = ShuffleSplit(n_splits=10)
print("shuffle Cross-Validation:")
scores = cross_validate(clas, ppgs, segment_labels, cv=shuffle_split, scoring=scoring, return_train_score=show_training)
for metric_name, score in list(scores.items())[2:]:
    # print(score)
    print(f"Mean {metric_name}: {score.mean()} (±{score.std()})")


# split Rpeak_intvs
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