import os
import scipy.io
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import glob
import preprocess_ecg
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense


af_models = glob.glob('model/af/**/*.mat')
non_af_models = glob.glob('model/non_af/*.mat')

signal_labels = []
parameters = []
signals = []
ecgs = []
labels = []
Rpeak_intvs = []
signal_length = 7500 # 30 secs
sample_rate = 250
sample_size = 250

# data init
for model in af_models:
    data = scipy.io.loadmat(model)
    signal_labels.append(data['labels'])
    parameters.append(data['parameters'])
    signals = data['signals'][0, 0]

    ecg_list = signals[1][0] # ['multileadECG'] I (may change when noise diff)
    ecg = [i[0] for i in ecg_list]
    # if len(ecg) < signal_length:
    #     print(len(ecg))
    ecg = np.array(ecg[0:signal_length])
    ecg = ecg.reshape(len(ecg))
    ecgs.append(ecg)

    labels.append(True)

for model in non_af_models:
    data = scipy.io.loadmat(model)
    signal_labels.append(data['labels'])
    parameters.append(data['parameters'])
    signals = data['signals'][0, 0]

    ecg_list = signals[1][0] # ['multileadECG'] I (may change when noise diff)
    ecg = [i[0] for i in ecg_list]
    # if len(ecg) < signal_length:
    #     print(len(ecg))
    ecg = np.array(ecg[0:signal_length])
    ecg = ecg.reshape(len(ecg))
    ecgs.append(ecg)

    labels.append(False)

# print(len(ecgs))


labels = np.array(labels)
labels = labels.reshape(len(labels), 1)


# ecg Rpeak intervals
for ecg in ecgs:
    ecg_index = np.arange(len(ecg))
    Rpeak_intv = preprocess_ecg.Rpeak_intervals(ecg, ecg_index)
    Rpeak_intvs.append(Rpeak_intv)

max_Rpeak_intv = max(len(arr) for arr in Rpeak_intvs)
Rpeak_intvs = [np.pad(Rpeak_intv, (0, max_Rpeak_intv - len(Rpeak_intv)), mode='constant') for Rpeak_intv in Rpeak_intvs]
Rpeak_intvs = np.array(Rpeak_intvs)
# print(Rpeak_intvs.shape)


# ecg instantaneous frequencies (time-dependent)
tdfs = np.array([preprocess_ecg.time_dependent_frequency(ecg, sample_rate) for ecg in ecgs])
print(tdfs.shape)

# ecg spectral entropies
ses = np.array([preprocess_ecg.spectral_entropy(ecg, sample_rate) for ecg in ecgs])
print(ses.shape)

# ecg sample arrays
# ecg_sample_labels = []
# ecg_samples = []
# for i, ecg in enumerate(ecgs):
#     for j in range(0, len(ecg), sample_size):
#         ecg_sample = ecg[j:j+sample_size]
#         if len(ecg_sample) == sample_size:
#             ecg_samples.append(ecg_sample)
#             ecg_sample_labels.append(labels[i])
# ecg_samples = np.array(ecg_samples)
# ecg_sample_labels = np.array(ecg_sample_labels)
# print(ecg_samples.shape, type(ecg_samples), type(ecg_samples[0]), type(ecg_samples[0, 0]), type(ecg_samples[0, 0, 0]))
# print(ecg_sample_labels.shape)


# splits
# Rpeak_intv_train, Rpeak_intv_test, Rpeak_intv_label_train, Rpeak_intv_label_test = train_test_split(Rpeak_intvs, labels, test_size=0.2, random_state=42)
# Rpeak_intv_train, Rpeak_intv_val, Rpeak_intv_label_train, Rpeak_intv_label_val = train_test_split(Rpeak_intv_train, Rpeak_intv_label_train, test_size=0.2, random_state=42)
# print((Rpeak_intv_train.shape), (Rpeak_intv_val.shape), (Rpeak_intv_test.shape))
# print((Rpeak_intv_label_train.shape), (Rpeak_intv_label_val.shape), (Rpeak_intv_label_test.shape))

tdf_train, tdf_test, tdf_label_train, tdf_label_test = train_test_split(tdfs, labels, test_size=0.2, random_state=42)
tdf_train, tdf_val, tdf_label_train, tdf_label_val = train_test_split(tdf_train, tdf_label_train, test_size=0.2, random_state=42)
print((tdf_train.shape), (tdf_val.shape), (tdf_test.shape))
print((tdf_label_train.shape), (tdf_label_val.shape), (tdf_label_test.shape))

se_train, se_test, se_label_train, se_label_test = train_test_split(ses, labels, test_size=0.2, random_state=42)
se_train, se_val, se_label_train, se_label_val = train_test_split(se_train, se_label_train, test_size=0.2, random_state=42)
print((se_train.shape), (se_val.shape), (se_test.shape))
print((se_label_train.shape), (se_label_val.shape), (se_label_test.shape))

# ecg_train, ecg_test, ecg_label_train, ecg_label_test = train_test_split(ecg_samples, ecg_sample_labels, test_size=0.2, random_state=42)
# ecg_train, ecg_val, ecg_label_train, ecg_label_val = train_test_split(ecg_train, ecg_label_train, test_size=0.2, random_state=42)
# print((ecg_train.shape), (ecg_val.shape), (ecg_test.shape))
# print((ecg_label_train.shape), (ecg_label_val.shape), (ecg_label_test.shape))


model = Sequential()
model.add(LSTM(units=32, input_shape=(len(tdfs[0]), 1)))
model.add(Dense(units=1, activation='sigmoid')) #T/F

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(Rpeak_intv_train, Rpeak_intv_label_train, validation_data=(Rpeak_intv_val, Rpeak_intv_label_val), epochs=100)

# loss, accuracy = model.evaluate(Rpeak_intv_test, Rpeak_intv_label_test)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

model.fit(tdf_train, tdf_label_train, validation_data=(tdf_val, tdf_label_val), epochs=100)

loss, accuracy = model.evaluate(tdf_test, tdf_label_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# model.fit(se_train, se_label_train, validation_data=(se_val, se_label_val), epochs=200)

# loss, accuracy = model.evaluate(se_test, se_label_test)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# model.fit(ecg_train, ecg_label_train, validation_data=(ecg_val, ecg_label_val), epochs=100)

# loss, accuracy = model.evaluate(ecg_test, ecg_label_test)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
