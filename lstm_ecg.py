import numpy as np
import preprocess_ecg
from sklearn.model_selection import train_test_split
import keras.metrics as km
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Bidirectional, Dropout

signal_length = 7500 # 30 secs (250Hz large_data mat)
min_freq = 5
max_freq = 40
length = 3750 # 30 secs (125Hz data_init csv)

def lstm_model(model):
    model.add(Bidirectional(LSTM(units=32)))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='sigmoid')) #T/F

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=preprocess_ecg.metrics)

    return model

print("lstm")
# get patient data
ecgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate = preprocess_ecg.data_init(min_freq, max_freq, length)
labels = np.array(segment_labels)

ffts, infs, ses = preprocess_ecg.feature_extraction(ecgs, sample_rate)
features = np.stack((infs, ses), axis=-1)
print(features.shape)

print("ecg signals")
model, feature_labels = preprocess_ecg.split_dataset(ecgs, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

intv_samples, sample_labels = preprocess_ecg.split_Rpeak_intvs(Rpeak_intvs, interval_labels)

print("Rpeak_intvs")
model, feature_labels = preprocess_ecg.split_dataset(intv_samples, sample_labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

print("ffts")
model, feature_labels = preprocess_ecg.split_dataset(ffts, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

print("infs")
model, feature_labels = preprocess_ecg.split_dataset(infs, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

print("ses")
model, feature_labels = preprocess_ecg.split_dataset(ses, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

print("infs & ses")
model, feature_labels = preprocess_ecg.split_dataset(features, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)


# get generated data
ecgs, labels, Rpeak_intvs, sample_rate = preprocess_ecg.large_data(signal_length)
ffts, infs, ses = preprocess_ecg.feature_extraction(ecgs, sample_rate)
features = np.stack((infs, ses), axis=-1)
print(features.shape)

print("ecg signals")
model, feature_labels = preprocess_ecg.split_dataset(ecgs, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

intv_samples, sample_labels = preprocess_ecg.split_Rpeak_intvs(Rpeak_intvs, labels)

print("Rpeak_intvs")
model, feature_labels = preprocess_ecg.split_dataset(intv_samples, sample_labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

print("ffts")
model, feature_labels = preprocess_ecg.split_dataset(ffts, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

print("infs")
model, feature_labels = preprocess_ecg.split_dataset(infs, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

print("ses")
model, feature_labels = preprocess_ecg.split_dataset(ses, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)

print("infs & ses")
model, feature_labels = preprocess_ecg.split_dataset(features, labels)
model = lstm_model(model)
preprocess_ecg.model_fit(model, feature_labels)