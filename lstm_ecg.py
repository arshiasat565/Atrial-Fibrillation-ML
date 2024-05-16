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
    ecg = preprocess_ecg.flatten_filter(ecg, 1, 40, sample_rate=sample_rate)
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
    ecg = preprocess_ecg.flatten_filter(ecg, 1, 40, sample_rate=sample_rate)
    ecgs.append(ecg)

    labels.append(False)

# print(len(ecgs))


labels = np.array(labels)
labels = labels.reshape(len(labels), 1)


# ecg instantaneous frequencies (time-dependent)
tdfs = np.array([preprocess_ecg.time_dependent_frequency(ecg, sample_rate) for ecg in ecgs])
tdf_mean = np.mean(tdfs)
tdf_std = np.std(tdfs)
tdfs = np.array([(x - tdf_mean) / tdf_std for x in tdfs])

# ecg spectral entropies
ses = np.array([preprocess_ecg.spectral_entropy(ecg, sample_rate) for ecg in ecgs])
se_mean = np.mean(ses)
se_std = np.std(ses)
ses = np.array([(x - se_mean) / se_std for x in ses])

# print(tdfs.shape, ses.shape)
features = np.stack((tdfs, ses), axis=-1)


# splits
feature_train, feature_test, feature_label_train, feature_label_test = train_test_split(features, labels, test_size=0.2)
feature_train, feature_val, feature_label_train, feature_label_val = train_test_split(feature_train, feature_label_train, test_size=0.2)
print((feature_train.shape), (feature_val.shape), (feature_test.shape))
print((feature_label_train.shape), (feature_label_val.shape), (feature_label_test.shape))


model = Sequential()
model.add(LSTM(units=32, input_shape=(len(features[0]), 2)))
model.add(Dense(units=1, activation='sigmoid')) #T/F

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])







model.fit(feature_train, feature_label_train, validation_data=(feature_val, feature_label_val), epochs=100, verbose=0)

loss, accuracy = model.evaluate(feature_test, feature_label_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


