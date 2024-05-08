import os
import scipy.io
import numpy as np
import glob
import preprocess_ppg
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense


af_models = glob.glob('model/af/**/*.mat')
non_af_models = glob.glob('model/non_af/*.mat')

signal_labels = []
parameters = []
signals = []
ppgs = []
labels = []
Rpeak_intv1s = []
Rpeak_intv2s = []
sample_size = 250

# data init
for model in af_models:
    data = scipy.io.loadmat(model)
    signal_labels.append(data['labels'])
    parameters.append(data['parameters'])
    signals = data['signals'][0, 0]

    ppg = np.array(signals['PPG']) # ['PPG'] 1&2
    ppgs.append(ppg)

    # Rpeaks from ppg
    ppg_index = np.arange(len(ppg))
    print(ppg.shape)
    Rpeak_intv1 = preprocess_ppg.Rpeak_intervals(ppg[0], ppg_index)
    Rpeak_intv2 = preprocess_ppg.Rpeak_intervals(ppg[1], ppg_index)
    Rpeak_intv1s.append(Rpeak_intv1)
    Rpeak_intv2s.append(Rpeak_intv2)
    labels.append(True)
    # print(len(signals['PPG'][0]))

for model in non_af_models:
    data = scipy.io.loadmat(model)
    signal_labels.append(data['labels'])
    parameters.append(data['parameters'])
    signals = data['signals'][0, 0]

    ppg = np.array(signals['PPG']) # ['PPG'] 1&2
    ppgs.append(ppg)

    # Rpeaks from ppg
    ppg_index = np.arange(len(ppg))
    print(ppg.shape)
    Rpeak_intv1 = preprocess_ppg.Rpeak_intervals(ppg[0], ppg_index)
    Rpeak_intv2 = preprocess_ppg.Rpeak_intervals(ppg[1], ppg_index)
    Rpeak_intv1s.append(Rpeak_intv1)
    Rpeak_intv2s.append(Rpeak_intv2)
    labels.append(False)
    # print(len(signals['PPG'][0]))

print(len(ppgs))

labels = np.array(labels)
labels = labels.reshape(len(labels), 1)

# ppg Rpeak intervals
max_Rpeak_intv1 = max(len(arr) for arr in Rpeak_intv1s)
Rpeak_intv1s = [np.pad(Rpeak_intv, (0, max_Rpeak_intv1 - len(Rpeak_intv)), mode='constant') for Rpeak_intv in Rpeak_intv1s]
Rpeak_intv1s = np.array(Rpeak_intv1s)
# print(Rpeak_intv1s.shape)
max_Rpeak_intv2 = max(len(arr) for arr in Rpeak_intv2s)
Rpeak_intv2s = [np.pad(Rpeak_intv, (0, max_Rpeak_intv2 - len(Rpeak_intv)), mode='constant') for Rpeak_intv in Rpeak_intv2s]
Rpeak_intv2s = np.array(Rpeak_intv2s)
# print(Rpeak_intv2s.shape)


# ppg sample arrays
ppg_sample_labels = []
ppg_samples = []
for i, ppg in enumerate(ppgs):
    for j in range(0, len(ppg), sample_size):
        ppg_sample = ppg[j:j+sample_size]
        if len(ppg_sample) == sample_size:
            ppg_samples.append(ppg_sample)
            ppg_sample_labels.append(labels[i])
ppg_samples = np.array(ppg_samples)
ppg_sample_labels = np.array(ppg_sample_labels)
# print(ppg_samples.shape, type(ppg_samples), type(ppg_samples[0]), type(ppg_samples[0, 0]), type(ppg_samples[0, 0, 0]))
# print(ppg_sample_labels.shape)


# splits
Rpeak_intv_train, Rpeak_intv_test, Rpeak_intv_label_train, Rpeak_intv_label_test = train_test_split(Rpeak_intvs, labels, test_size=0.2, random_state=42)
Rpeak_intv_train, Rpeak_intv_val, Rpeak_intv_label_train, Rpeak_intv_label_val = train_test_split(Rpeak_intv_train, Rpeak_intv_label_train, test_size=0.2, random_state=42)

ppg_train, ppg_test, ppg_label_train, ppg_label_test = train_test_split(ppgs, labels, test_size=0.2, random_state=42)
ppg_train, ppg_val, ppg_label_train, ppg_label_val = train_test_split(ppg_train, ppg_label_train, test_size=0.2, random_state=42)

# print((Rpeak_intv_train.shape), (Rpeak_intv_val.shape), (Rpeak_intv_test.shape))
# print((Rpeak_intv_label_train.shape), (Rpeak_intv_label_val.shape), (Rpeak_intv_label_test.shape))

# print((ppg_train.shape), (ppg_val.shape), (ppg_test.shape))
# print((ppg_label_train.shape), (ppg_label_val.shape), (ppg_label_test.shape))


model = Sequential()
model.add(LSTM(units=32, input_shape=(sample_size, 1)))
model.add(Dense(units=1, activation='sigmoid')) #T/F

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(Rpeak_intv_train, Rpeak_intv_label_train, validation_data=(Rpeak_intv_val, Rpeak_intv_label_val), epochs=100)

loss, accuracy = model.evaluate(Rpeak_intv_test, Rpeak_intv_label_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# model.fit(ppg_train, ppg_label_train, validation_data=(ppg_val, ppg_label_val), epochs=100)

# loss, accuracy = model.evaluate(ppg_test, ppg_label_test)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
