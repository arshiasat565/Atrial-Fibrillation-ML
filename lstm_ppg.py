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

def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

metrics = [
    'accuracy', 'precision', 'recall', km.AUC(curve='ROC')
]

# # get patient data
# print("patient")
# ppgs, times, Rpeak_intvs, segment_labels, interval_labels, sample_rate = preprocess_ppg.data_init(min_freq, max_freq, length)
# labels = np.array(segment_labels)

# # feature extraction
# # ppg instantaneous frequencies (time-dependent)
# infs = np.array([preprocess_ppg.time_dependent_frequency(ppg, sample_rate) for ppg in ppgs])
# inf_mean = np.mean(infs)
# inf_std = np.std(infs)
# infs = np.array([(x - inf_mean) / inf_std for x in infs])

# # ppg spectral entropies
# ses = np.array([preprocess_ppg.spectral_entropy(ppg, sample_rate) for ppg in ppgs])
# se_mean = np.mean(ses)
# se_std = np.std(ses)
# ses = np.array([(x - se_mean) / se_std for x in ses])

# print(infs.shape, ses.shape)
# features = np.stack((infs, ses), axis=-1)
# print(features.shape)

# get generated data
print("generated")
ppgs, labels, Rpeak_intvs, sample_rate = preprocess_ppg.large_data(signal_length)
ppgs, labels = np.array(ppgs), np.array(labels)
print(ppgs.shape, labels.shape)

# feature extraction
# ppg instantaneous frequencies (time-dependent)
inf1s = np.array([preprocess_ppg.time_dependent_frequency(ppg[0], sample_rate) for ppg in ppgs])
inf_mean = np.mean(inf1s)
inf_std = np.std(inf1s)
inf1s = np.array([(x - inf_mean) / inf_std for x in inf1s])
inf2s = np.array([preprocess_ppg.time_dependent_frequency(ppg[1], sample_rate) for ppg in ppgs])
inf_mean = np.mean(inf2s)
inf_std = np.std(inf2s)
inf2s = np.array([(x - inf_mean) / inf_std for x in inf2s])

# ppg spectral entropies
se1s = np.array([preprocess_ppg.spectral_entropy(ppg[0], sample_rate) for ppg in ppgs])
se_mean = np.mean(se1s)
se_std = np.std(se1s)
se1s = np.array([(x - se_mean) / se_std for x in se1s])
se2s = np.array([preprocess_ppg.spectral_entropy(ppg[1], sample_rate) for ppg in ppgs])
se_mean = np.mean(se2s)
se_std = np.std(se2s)
se2s = np.array([(x - se_mean) / se_std for x in se2s])

print(inf1s.shape, inf2s.shape, se1s.shape, se2s.shape)
infs = np.concatenate((inf1s, inf2s))
ses = np.concatenate((se1s, se2s))
features = np.stack((infs, ses), axis=-1)
print(features.shape)

labels = np.concatenate((labels, labels))

# splits
feature_train, feature_test, feature_label_train, feature_label_test = train_test_split(features, labels, test_size=0.2)
feature_train, feature_val, feature_label_train, feature_label_val = train_test_split(feature_train, feature_label_train, test_size=0.2)
print((feature_train.shape), (feature_val.shape), (feature_test.shape))
print((feature_label_train.shape), (feature_label_val.shape), (feature_label_test.shape))


model = Sequential()
model.add(Input(shape=(len(ppgs[0]), 2)))
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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

# fit several times on the same split
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
