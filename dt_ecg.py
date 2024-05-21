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

min_freq = 5
max_freq = 40
start = 0
length = 150000 # 20 mins
sample_rate = 125

def cross_val(clas, ecgs, labels, scoring, return_train_score):
    print("Cross-Validation:")
    scores = cross_validate(clas, ecgs, labels, scoring=scoring, return_train_score=return_train_score)
    score_list = list(scores.items())[2:]
    for metric_name, score in score_list:
        print(f"Mean {metric_name}: {score.mean():.2f} (±{score.std():.2f})")


ecgs, times, Rpeak_intvs, segment_labels, interval_labels = preprocess_ecg.data_init(min_freq, max_freq, length, sample_rate)


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
            sample_labels.append(interval_labels[i])
print("sample count:", len(intv_samples))

intv_samples_train, intv_samples_test, labels_train, labels_test = train_test_split(intv_samples, sample_labels, test_size=0.2)


dt_classifier = DecisionTreeClassifier(criterion='gini', splitter='best')
dt_classifier.fit(intv_samples_train, labels_train)

# Predict labels for testing data
labels_pred = dt_classifier.predict(intv_samples_test)

# Evaluate performance
accuracy = accuracy_score(labels_test, labels_pred)
report = classification_report(labels_test, labels_pred)
print(f"{dt_classifier}")
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

cross_val(dt_classifier, intv_samples, sample_labels, scoring, show_training)


rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(intv_samples_train, labels_train)

# Predict labels for testing data
labels_pred = rf_classifier.predict(intv_samples_test)

# Evaluate performance
accuracy = accuracy_score(labels_test, labels_pred)
report = classification_report(labels_test, labels_pred)
print(f"\n{rf_classifier}")
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

cross_val(rf_classifier, intv_samples, sample_labels, scoring, show_training)