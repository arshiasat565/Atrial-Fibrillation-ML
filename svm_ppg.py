import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import lab_funcs_ppg
import preprocess_ppg
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# scoring metrics
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
sample_rate = 125

def cross_val(clas, ppgs, labels, cv, scoring, return_train_score):
    print(f"{cv} Cross-Validation:")
    scores = cross_validate(clas, ppgs, labels, cv=cv, scoring=scoring, return_train_score=return_train_score)
    score_list = list(scores.items())[2:]
    for metric_name, score in score_list:
        print(f"Mean {metric_name}: {score.mean():.2f} (±{score.std():.2f})")

# get patient data
ppgs, times, Rpeak_intvs, segment_labels, interval_labels = preprocess_ppg.data_init(min_freq, max_freq, length, sample_rate)

# length_sec = length / sample_rate
# # split ppg
# print(f"\nBy {length_sec}s ppg samples")
# print("sample count:", len(ppgs))

# # 10 fold cross validation svm, use ppg samples
# # kfold
# kf = KFold(n_splits=10, shuffle=True)
# cross_val(clas, ppgs, segment_labels, kf, scoring, show_training) #67%acc

# # shuffle
# shuffle_split = ShuffleSplit(n_splits=10)
# cross_val(clas, ppgs, segment_labels, shuffle_split, scoring, show_training) #67%acc

# split Rpeak_intvs
print("\nBy Rpeak_intv samples")
sample_size = 10
intv_samples = []
sample_labels = []

for i, Rpeak_intv in enumerate(Rpeak_intvs):
    for j in range(0, len(Rpeak_intv), sample_size):
        intv_sample = Rpeak_intv[j:j+sample_size]
        if len(intv_sample) == sample_size:
            intv_samples.append(intv_sample)
            sample_labels.append(interval_labels[i])
print("sample count:", len(intv_samples))

# 10 fold cross validation svm, use Rpeak_intv samples
clas = svm.SVC(kernel="rbf", probability=True, C=1, gamma=0.001) #optimised rbf params
# kfold
kf = KFold(n_splits=10, shuffle=True)
cross_val(clas, intv_samples, sample_labels, kf, scoring, show_training) #90%acc w rbf params

# shuffle
shuffle_split = ShuffleSplit(n_splits=10)
cross_val(clas, intv_samples, sample_labels, shuffle_split, scoring, show_training) #90%acc w rbf params

# # diff train test split changes accuracy by ±1%

# split_range = np.linspace(0.1, 0.9, 9)
# for split in split_range:
#     print("train_test_split:", split)
#     ppgs_train, ppgs_test, segment_labels_train, segment_labels_test = train_test_split(intv_samples, sample_labels, test_size=split)
#     clas.fit(ppgs_train, segment_labels_train)
#     segment_labels_pred = clas.predict(ppgs_test)
#     accuracy = accuracy_score(segment_labels_test, segment_labels_pred)
#     print(accuracy)
