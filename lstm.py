import os
import scipy.io
import numpy as np
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
ppgs = []
labels = []
Rpeak_intvs = []
sample_size = 250

# data init
for model in af_models:
    data = scipy.io.loadmat(model)
    signal_labels.append(data['labels'])
    parameters.append(data['parameters'])
    signals = data['signals'][0, 0]

    ecg_list = signals[1][0] # ['multileadECG'] I (may change when noise diff)
    ecg = [i[0] for i in ecg_list]
    ecg = np.array(ecg)
    ecgs.append(ecg)

    ecg_index = np.arange(len(ecg))
    new_ecg = ecg.reshape(len(ecg))
    Rpeak_intv = preprocess_ecg.Rpeak_intervals(new_ecg, ecg_index)
    Rpeak_intvs.append(Rpeak_intv)

    ppg = np.array(signals['PPG']) # ['PPG'] 1&2
    ppgs.append(ppg)
    labels.append(True)
    # print(len(ecg), len(signals['PPG'][0]))

for model in non_af_models:
    data = scipy.io.loadmat(model)
    signal_labels.append(data['labels'])
    parameters.append(data['parameters'])
    signals = data['signals'][0, 0]

    ecg_list = signals[1][0] # ['multileadECG'] I (may change when noise diff)
    ecg = [i[0] for i in ecg_list]
    ecg = np.array(ecg)
    ecgs.append(ecg)

    # Rpeaks from ecg
    ecg_index = np.arange(len(ecg))
    new_ecg = ecg.reshape(len(ecg))
    Rpeak_intv = preprocess_ecg.Rpeak_intervals(new_ecg, ecg_index)
    Rpeak_intvs.append(Rpeak_intv)

    ppg = np.array(signals['PPG']) # ['PPG'] 1&2
    ppgs.append(ppg)
    labels.append(False)
    # print(len(ecg), len(signals['PPG'][0]))

# print(len(ecgs))

labels = np.array(labels)
labels = labels.reshape(len(labels), 1)

max_Rpeak_intv = max(len(arr) for arr in Rpeak_intvs)
Rpeak_intvs = [np.pad(Rpeak_intv, (0, max_Rpeak_intv - len(Rpeak_intv)), mode='constant') for Rpeak_intv in Rpeak_intvs]
Rpeak_intvs = np.array(Rpeak_intvs)
# print(Rpeak_intvs.shape)


# ecg sample arrays
ecg_sample_labels = []
ecg_samples = []
for i, ecg in enumerate(ecgs):
    for j in range(0, len(ecg), sample_size):
        ecg_sample = ecg[j:j+sample_size]
        if len(ecg_sample) == sample_size:
            ecg_samples.append(ecg_sample)
            ecg_sample_labels.append(labels[i])
ecg_samples = np.array(ecg_samples)
ecg_sample_labels = np.array(ecg_sample_labels)
# print(ecg_samples.shape, type(ecg_samples), type(ecg_samples[0]), type(ecg_samples[0, 0]), type(ecg_samples[0, 0, 0]))
# print(ecg_sample_labels.shape)

# ppg sample arrays
# min_ppg_size = min(len(ppg[0]) for ppg in ppgs)
# ppgs = [np.transpose(ppg[:, :min_ppg_size]) for ppg in ppgs]
# ppgs = np.array(ppgs)
# print(ppgs.shape, type(ppgs), type(ppgs[0]), type(ppgs[0, 0]), type(ppgs[0, 0, 0]))


# splits
Rpeak_intv_train, Rpeak_intv_test, Rpeak_intv_label_train, Rpeak_intv_label_test = train_test_split(Rpeak_intvs, labels, test_size=0.2, random_state=42)
Rpeak_intv_train, Rpeak_intv_val, Rpeak_intv_label_train, Rpeak_intv_label_val = train_test_split(Rpeak_intv_train, Rpeak_intv_label_train, test_size=0.2, random_state=42)

ecg_train, ecg_test, ecg_label_train, ecg_label_test = train_test_split(ecg_samples, ecg_sample_labels, test_size=0.2, random_state=42)
ecg_train, ecg_val, ecg_label_train, ecg_label_val = train_test_split(ecg_train, ecg_label_train, test_size=0.2, random_state=42)

ppg_train, ppg_test, ppg_label_train, ppg_label_test = train_test_split(ppgs, labels, test_size=0.2, random_state=42)
ppg_train, ppg_val, ppg_label_train, ppg_label_val = train_test_split(ppg_train, ppg_label_train, test_size=0.2, random_state=42)

# print((Rpeak_intv_train.shape), (Rpeak_intv_val.shape), (Rpeak_intv_test.shape))
# print((Rpeak_intv_label_train.shape), (Rpeak_intv_label_val.shape), (Rpeak_intv_label_test.shape))

# print((ecg_train.shape), (ecg_val.shape), (ecg_test.shape))
# print((ecg_label_train.shape), (ecg_label_val.shape), (ecg_label_test.shape))
# print((ppg_train.shape), (ppg_val.shape), (ppg_test.shape))


model = Sequential()
model.add(LSTM(units=32, input_shape=(sample_size, 1)))
model.add(Dense(units=1, activation='sigmoid')) #T/F

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(Rpeak_intv_train, Rpeak_intv_label_train, validation_data=(Rpeak_intv_val, Rpeak_intv_label_val), epochs=100)

# model.fit(ecg_train, ecg_label_train, validation_data=(ecg_val, ecg_label_val), epochs=100)

loss, accuracy = model.evaluate(ecg_test, ecg_label_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
