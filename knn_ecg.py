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
from sklearn.neighbors import KNeighborsClassifier
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


# get patient data
ecgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate = preprocess_ecg.data_init(min_freq, max_freq, length)

# feature extraction
ffts = np.array([preprocess_ecg.fft(ecg, sample_rate) for ecg in ecgs])

ffts_train, ffts_test, labels_train, labels_test = train_test_split(ffts, segment_labels, test_size=0.2)


# setup k nearest neighbour model
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(ffts_train, labels_train)

# Predict labels for testing data
labels_pred = knn_classifier.predict(ffts_test)

# Evaluate performance
accuracy = accuracy_score(labels_test, labels_pred)
report = classification_report(labels_test, labels_pred)

print(f"{knn_classifier}")
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

cross_val(knn_classifier, ffts, segment_labels, scoring, show_training)