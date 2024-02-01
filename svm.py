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
labels = []

for csv in af_csv:
    df = pd.read_csv(csv)
    ecg = basic.flatten_filter(df.ECG)
    # dfs.append((ecg, True))
    if not any(np.isnan(value) for value in ecg):
        ecgs.append(ecg)
        labels.append(True)

print('non')
for csv in non_af_csv:
    df = pd.read_csv(csv)
    ecg = basic.flatten_filter(df.ECG)
    # dfs.append((ecg, False))
    if not any(np.isnan(value) for value in ecg):
        ecgs.append(ecg)
        labels.append(False)

print(len(ecgs), len(labels))

ecg_train, ecg_test, label_train, label_test = train_test_split(ecgs, labels, test_size=0.2, random_state=8964)

clas = svm.SVC(kernel="linear")

print(len(ecg_train), len(label_train))

clas.fit(ecg_train, label_train)

label_pred = clas.predict(ecg_test)

acc = accuracy_score(label_test, label_pred)
print(acc)