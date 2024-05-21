import numpy as np
import preprocess_ecg
from sklearn.model_selection import train_test_split
import keras.metrics as km
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout

signal_length = 7500 # 30 secs (250Hz large_data mat)
min_freq = 5
max_freq = 40
length = 3750 # 30 secs (125Hz data_init csv)

def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

metrics = [
    'accuracy', 'precision', 'recall', km.AUC(curve='ROC')
]


# # get patient data
# print("patient")
# ecgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate = preprocess_ecg.data_init(min_freq, max_freq, length)
# labels = np.array(segment_labels)

# get generated data
print("generated")
ecgs, labels, sample_rate = preprocess_ecg.large_data(signal_length)

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
model.add(Input(shape=(len(features[0]), 2)))
print("32")
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
print("64")
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation='sigmoid')) #T/F

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

accs = []
losses = []
results = []
for i in range(10):
    model.fit(feature_train, feature_label_train, validation_data=(feature_val, feature_label_val), epochs=100, verbose=0)

    result = model.evaluate(feature_test, feature_label_test)
    result.append(f1_score(result[2], result[3]))
    # print(result)
    results.append(result)

metric_names = np.concatenate((['loss'], metrics, ['f1_score']))
np.set_printoptions(suppress=True)
avg_results = np.average(results, axis=0)
print("Average metrics:")
for name, value in zip(metric_names, avg_results):
    print(f'{name}: {value:.4f}')

#     loss, accuracy = model.evaluate(feature_test, feature_label_test)
#     print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
#     losses.append(loss)
#     accs.append(accuracy)
# print(np.average(losses), np.average(accs))
