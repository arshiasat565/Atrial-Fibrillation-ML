import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import sklearn as skl
import glob
import lab_funcs
import basic

af_csv = glob.glob('mimic_perform_af_csv/*.csv')
non_af_csv = glob.glob('mimic_perform_non_af_csv/*.csv')

column_names = [
    "Time",
    "PPG",
    "ECG",
    "resp"
]

af_dfs = []
non_af_dfs = []

for csv in af_csv:
    df = pd.read_csv(csv)
    ecg = basic.flatten_filter(df.ECG)
    af_dfs.append((ecg, True))

print(af_dfs)

print('non')
for csv in non_af_csv:
    df = pd.read_csv(csv)
    ecg = basic.flatten_filter(df.ECG)
    non_af_dfs.append((ecg, False))

print(non_af_dfs)