import numpy as np
import preprocess_ecg
from sklearn.model_selection import train_test_split
import keras.api.metrics as km
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Input, Bidirectional, Dropout
from keras.api.callbacks import EarlyStopping, ModelCheckpoint

signal_length = 7500 # 30 secs (250Hz large_data mat)
min_freq = 5
max_freq = 40
length = 3750 # 30 secs (125Hz data_init csv)

def lstm_model(features):
    features = np.array(features)
    if features.ndim > 2:
        num = features.shape[2]
    else:
        num = 1
    model = Sequential()
    model.add(Input(shape=(features.shape[1], num)))
    print("64")
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Dropout(0.25))
    print("32")
    model.add(Bidirectional(LSTM(units=32)))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='sigmoid')) #T/F

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=preprocess_ecg.metrics)

    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
callbacks = []

print("lstm")
# get patient data
ecgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate = preprocess_ecg.data_init(min_freq, max_freq, length)
labels = np.array(segment_labels)

print("ecg signals")
preprocess_ecg.model_fit(lstm_model, callbacks, ecgs, labels)

intv_samples, sample_labels = preprocess_ecg.split_Rpeak_intvs(Rpeak_intvs, interval_labels)

print("Rpeak_intvs")
preprocess_ecg.model_fit(lstm_model, callbacks, intv_samples, sample_labels)

ffts, infs, ses = preprocess_ecg.feature_extraction(ecgs, sample_rate)
features = np.stack((infs, ses), axis=-1)
print(features.shape)

print("ffts")
preprocess_ecg.model_fit(lstm_model, callbacks, ffts, labels)

print("infs")
preprocess_ecg.model_fit(lstm_model, callbacks, infs, labels)

print("ses")
preprocess_ecg.model_fit(lstm_model, callbacks, ses, labels)

print("infs & ses")
preprocess_ecg.model_fit(lstm_model, callbacks, features, labels)


# # get generated data
# ecgs, labels, Rpeak_intvs, sample_rate = preprocess_ecg.large_data(signal_length)

# # ecgs, labels, Rpeak_intvs, sample_rate = preprocess_ecg.demo_data(2500) # 10 secs (250Hz large_data mat)
# ffts, infs, ses = preprocess_ecg.feature_extraction(ecgs, sample_rate)
# features = np.stack((infs, ses), axis=-1)
# print(features.shape)

# print("ecg signals")
# model, feature_labels = preprocess_ecg.split_dataset(ecgs, labels)
# model = lstm_model(model)
# preprocess_ecg.model_fit(model, feature_labels)

# intv_samples, sample_labels = preprocess_ecg.split_Rpeak_intvs(Rpeak_intvs, labels)

# print("Rpeak_intvs")
# model, feature_labels = preprocess_ecg.split_dataset(intv_samples, sample_labels)
# model = lstm_model(model)
# preprocess_ecg.model_fit(model, feature_labels)

# print("ffts")
# model, feature_labels = preprocess_ecg.split_dataset(ffts, labels)
# model = lstm_model(model)
# preprocess_ecg.model_fit(model, feature_labels)

# print("infs")
# model, feature_labels = preprocess_ecg.split_dataset(infs, labels)
# model = lstm_model(model)
# preprocess_ecg.model_fit(model, feature_labels)

# print("ses")
# model, feature_labels = preprocess_ecg.split_dataset(ses, labels)
# model = lstm_model(model)
# preprocess_ecg.model_fit(model, feature_labels)

# print("infs & ses")
# model, feature_labels = preprocess_ecg.split_dataset(features, labels)
# model = lstm_model(model)
# preprocess_ecg.model_fit(model, feature_labels)