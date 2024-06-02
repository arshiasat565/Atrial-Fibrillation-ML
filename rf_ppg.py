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
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}
show_training = False

signal_length = 7500 # 30 secs (250Hz large_data mat)
min_freq = 0.5
max_freq = 5
start = 0
length = 3750 # 30 secs (125Hz data_init csv)

def cross_val(clas, ppgs, labels, scoring, return_train_score):
    print("Cross-Validation:")
    scores = cross_validate(clas, ppgs, labels, scoring=scoring, return_train_score=return_train_score)
    score_list = list(scores.items())[2:]
    for metric_name, score in score_list:
        print(f"Mean {metric_name}: {score.mean():.2f} (±{score.std():.2f})")

print("rf")
# setup random forest model
clas = RandomForestClassifier(n_estimators=100)

# # get patient data
# ppgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate = preprocess_ppg.data_init(min_freq, max_freq, length)

# # split ppg
# length_sec = length / sample_rate
# print(f"\nBy {length_sec}s ppg samples")
# print("sample count:", len(ppgs))
# # 10 fold cross validation dt, use ppg samples
# cross_val(clas, ppgs, segment_labels, scoring, show_training)

# intv_samples, sample_labels = preprocess_ppg.split_Rpeak_intvs(Rpeak_intvs, interval_labels)

# # 10 fold cross validation dt, use Rpeak_intv samples
# cross_val(clas, intv_samples, sample_labels, scoring, show_training)

# ffts, infs, ses = preprocess_ppg.feature_extraction_db(ppgs, sample_rate)
# features = np.stack((infs, ses), axis=-1)
# print(features.shape)

# print("ffts")
# cross_val(clas, ffts, segment_labels, scoring, show_training)

# print("infs")
# cross_val(clas, infs, segment_labels, scoring, show_training)

# print("ses")
# cross_val(clas, ses, segment_labels, scoring, show_training)


# get generated data
ppgs, labels, Rpeak_intvs, interval_labels, sample_rate = preprocess_ppg.large_data(signal_length)
labels = labels.ravel()
flat_ppgs = np.concatenate((ppgs[:, 0, :], ppgs[:, 1, :]), axis=0)
flat_labels = np.concatenate((labels, labels), axis=0)

# 10 fold cross validation dt, use ppg samples
cross_val(clas, flat_ppgs, flat_labels, scoring, show_training)

intv_samples, sample_labels = preprocess_ppg.split_Rpeak_intvs(Rpeak_intvs, interval_labels)

# 10 fold cross validation dt, use Rpeak_intv samples
cross_val(clas, intv_samples, sample_labels, scoring, show_training)

ffts, infs, ses, labels = preprocess_ppg.feature_extraction_gen(ppgs, labels, sample_rate)
features = np.stack((infs, ses), axis=-1)
print(features.shape)

print("ffts")
cross_val(clas, ffts, labels, scoring, show_training)

print("infs")
cross_val(clas, infs, labels, scoring, show_training)

print("ses")
cross_val(clas, ses, labels, scoring, show_training)