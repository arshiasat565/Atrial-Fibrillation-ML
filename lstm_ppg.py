import numpy as np
import preprocess_ppg
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.metrics as km
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Bidirectional, Dropout

signal_length = 7500 # 30 secs (250Hz large_data mat)
min_freq = 0.5
max_freq = 5
length = 3750 # 30 secs (125Hz data_init csv)

def lstm_model(model):
    # print("128")
    # model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    # model.add(Dropout(0.25))
    print("64")
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Dropout(0.25))
    print("32")
    model.add(Bidirectional(LSTM(units=32)))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='sigmoid')) #T/F

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=preprocess_ppg.metrics)

    return model

print("lstm")
# get patient data
ppgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate = preprocess_ppg.data_init(min_freq, max_freq, length)
labels = np.array(segment_labels)

ffts, infs, ses = preprocess_ppg.feature_extraction_db(ppgs, sample_rate)
features = np.stack((infs, ses), axis=-1)
print(features.shape)

# print("ppg signals")
# model, feature_labels = preprocess_ppg.split_dataset(ppgs, labels)
# model = lstm_model(model)
# preprocess_ppg.model_fit(model, feature_labels)

# intv_samples, sample_labels = preprocess_ppg.split_Rpeak_intvs(Rpeak_intvs, interval_labels)

# print("Rpeak_intvs")
# model, feature_labels = preprocess_ppg.split_dataset(intv_samples, sample_labels)
# model = lstm_model(model)
# preprocess_ppg.model_fit(model, feature_labels)

print("ffts")
model, feature_labels = preprocess_ppg.split_dataset(ffts, labels)
model = lstm_model(model)
preprocess_ppg.model_fit(model, feature_labels)

print("infs")
model, feature_labels = preprocess_ppg.split_dataset(infs, labels)
model = lstm_model(model)
preprocess_ppg.model_fit(model, feature_labels)

print("ses")
model, feature_labels = preprocess_ppg.split_dataset(ses, labels)
model = lstm_model(model)
preprocess_ppg.model_fit(model, feature_labels)

print("infs & ses")
model, feature_labels = preprocess_ppg.split_dataset(features, labels)
model = lstm_model(model)
preprocess_ppg.model_fit(model, feature_labels)


# get generated data
ppgs, labels, Rpeak_intvs, interval_labels, sample_rate = preprocess_ppg.large_data(signal_length)

# print("ppg signals")
# model, feature_labels = preprocess_ppg.split_dataset(ppgs, labels)
# model = lstm_model(model)
# preprocess_ppg.model_fit(model, feature_labels)

# intv_samples, sample_labels = preprocess_ppg.split_Rpeak_intvs(Rpeak_intvs, interval_labels)

# print("Rpeak_intvs")
# model, feature_labels = preprocess_ppg.split_dataset(intv_samples, sample_labels)
# model = lstm_model(model)
# preprocess_ppg.model_fit(model, feature_labels)

ffts, infs, ses, labels = preprocess_ppg.feature_extraction_gen(ppgs, labels, sample_rate)
features = np.stack((infs, ses), axis=-1)
print(features.shape)

print("ffts")
model, feature_labels = preprocess_ppg.split_dataset(ffts, labels)
model = lstm_model(model)
preprocess_ppg.model_fit(model, feature_labels)

print("infs")
model, feature_labels = preprocess_ppg.split_dataset(infs, labels)
model = lstm_model(model)
preprocess_ppg.model_fit(model, feature_labels)

print("ses")
model, feature_labels = preprocess_ppg.split_dataset(ses, labels)
model = lstm_model(model)
preprocess_ppg.model_fit(model, feature_labels)

print("infs & ses")
model, feature_labels = preprocess_ppg.split_dataset(features, labels)
model = lstm_model(model)
preprocess_ppg.model_fit(model, feature_labels)