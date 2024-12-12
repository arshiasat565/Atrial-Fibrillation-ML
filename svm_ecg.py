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
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, hinge_loss

# scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score),
    'hinge_loss': make_scorer(hinge_loss)
}
show_training = False

signal_length = 7500 # 30 secs (250Hz large_data mat)
min_freq = 5
max_freq = 40
start = 0
length = 3750 # 30 secs (125Hz data_init csv)

shuffle_split = ShuffleSplit(n_splits=10)

def cross_val(clas, ecgs, labels, scoring, return_train_score):
    print("Cross-Validation:")
    scores = cross_validate(clas, ecgs, labels, cv=shuffle_split, scoring=scoring, return_train_score=return_train_score)
    score_list = list(scores.items())[2:]
    for metric_name, score in score_list:
        print(f"Mean {metric_name}: {score.mean():.2f} (±{score.std():.2f})")

print("svm")
# get patient data
ecgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate = preprocess_ecg.data_init(min_freq, max_freq, length)
labels = np.array(segment_labels)

clas = svm.SVC(kernel="rbf", probability=True, C=1, gamma=0.001) #optimised rbf params C:[1, 10, 100]
length_sec = length / sample_rate
# split ecg
print(f"\nBy {length_sec}s ecg samples")
print("sample count:", len(ecgs))

# 10 fold cross validation svm, use ecg samples
cross_val(clas, ecgs, segment_labels, scoring, show_training) #67%acc

intv_samples, sample_labels = preprocess_ecg.split_Rpeak_intvs(Rpeak_intvs, interval_labels)

# 10 fold cross validation svm, use Rpeak_intv samples
cross_val(clas, intv_samples, sample_labels, scoring, show_training) #90%acc w rbf params

ffts, infs, ses = preprocess_ecg.feature_extraction(ecgs, sample_rate)
features = np.column_stack((infs, ses))
print(features.shape)

print("ffts")
cross_val(clas, ffts, labels, scoring, show_training)

print("infs")
cross_val(clas, infs, labels, scoring, show_training)

print("ses")
cross_val(clas, ses, labels, scoring, show_training)

print("infs & ses")
cross_val(clas, features, labels, scoring, show_training)


# # diff train test split changes accuracy by ±1%

# split_range = np.linspace(0.1, 0.9, 9)
# for split in split_range:
#     print("train_test_split:", split)
#     ecgs_train, ecgs_test, segment_labels_train, segment_labels_test = train_test_split(intv_samples, sample_labels, test_size=split)
#     clas.fit(ecgs_train, segment_labels_train)
#     segment_labels_pred = clas.predict(ecgs_test)
#     accuracy = accuracy_score(segment_labels_test, segment_labels_pred)
#     print(accuracy)

# ecgs, labels, Rpeak_intvs, sample_rate = preprocess_ecg.large_data(signal_length)
# # ecgs, labels, Rpeak_intvs, sample_rate = preprocess_ecg.demo_data(2500) # 10 secs (250Hz large_data mat)
# clas = svm.SVC(kernel="rbf", probability=True, C=1, gamma=0.001) #TODO optimised rbf params C:[1, 10, 100]
# length_sec = signal_length / sample_rate
# # split ecg
# print(f"\nBy {length_sec}s ecg samples")
# print("sample count:", len(ecgs))

# # 10 fold cross validation svm, use ecg samples
# shuffle_split = ShuffleSplit(n_splits=10)
# cross_val(clas, ecgs, labels, scoring, show_training) #TODO 67%acc

# intv_samples, sample_labels = preprocess_ecg.split_Rpeak_intvs(Rpeak_intvs, labels)

# # 10 fold cross validation svm, use Rpeak_intv samples
# cross_val(clas, intv_samples, sample_labels, scoring, show_training) #TODO 90%acc w rbf params

# ffts, infs, ses = preprocess_ecg.feature_extraction(ecgs, sample_rate)
# features = np.column_stack((infs, ses))
# print(features.shape)

# print("ffts")
# cross_val(clas, ffts, labels, scoring, show_training)

# print("infs")
# cross_val(clas, infs, labels, scoring, show_training)

# print("ses")
# cross_val(clas, ses, labels, scoring, show_training)

# print("infs & ses")
# cross_val(clas, features, labels, scoring, show_training)