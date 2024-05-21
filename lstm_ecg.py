import numpy as np
import preprocess_ecg
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Bidirectional, Dropout

signal_length = 7500 # 30 secs (large_data mat)
min_freq = 5
max_freq = 40
length = 3750 # 30 secs (data_init csv)
sample_rate = 250


ecgs, times, Rpeak_intvs, segment_labels, interval_labels = preprocess_ecg.data_init(min_freq, max_freq, length, sample_rate)
segment_labels = np.array(segment_labels)
# ecgs, labels = preprocess_ecg.large_data(signal_length, sample_rate)

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

print(tdfs.shape, ses.shape)
features = np.stack((tdfs, ses), axis=-1)


# splits
feature_train, feature_test, feature_label_train, feature_label_test = train_test_split(features, segment_labels, test_size=0.2)
feature_train, feature_val, feature_label_train, feature_label_val = train_test_split(feature_train, feature_label_train, test_size=0.2)
print((feature_train.shape), (feature_val.shape), (feature_test.shape))
print((feature_label_train.shape), (feature_label_val.shape), (feature_label_test.shape))


model = Sequential()
model.add(Input(shape=(len(features[0]), 2))) #99% acc, <0.1 loss
model.add(Bidirectional(LSTM(units=32)))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation='sigmoid')) #T/F

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

accs = []
losses = []
for i in range(10):
    model.fit(feature_train, feature_label_train, validation_data=(feature_val, feature_label_val), epochs=100, verbose=0)

    loss, accuracy = model.evaluate(feature_test, feature_label_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    losses.append(loss)
    accs.append(accuracy)
print(np.average(losses), np.average(accs))
