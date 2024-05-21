import numpy as np
import preprocess_ppg
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout

signal_length = 7500 # 30 secs
sample_rate = 250

ppgs, labels = preprocess_ppg.large_data(signal_length, sample_rate)

# ppg instantaneous frequencies (time-dependent)
tdf1s = np.array([preprocess_ppg.time_dependent_frequency(ppg[0], sample_rate) for ppg in ppgs])
tdf_mean = np.mean(tdf1s)
tdf_std = np.std(tdf1s)
tdf1s = np.array([(x - tdf_mean) / tdf_std for x in tdf1s])
tdf2s = np.array([preprocess_ppg.time_dependent_frequency(ppg[1], sample_rate) for ppg in ppgs])
tdf_mean = np.mean(tdf2s)
tdf_std = np.std(tdf2s)
tdf2s = np.array([(x - tdf_mean) / tdf_std for x in tdf2s])

# ppg spectral entropies
se1s = np.array([preprocess_ppg.spectral_entropy(ppg[0], sample_rate) for ppg in ppgs])
se_mean = np.mean(se1s)
se_std = np.std(se1s)
se1s = np.array([(x - se_mean) / se_std for x in se1s])
se2s = np.array([preprocess_ppg.spectral_entropy(ppg[1], sample_rate) for ppg in ppgs])
se_mean = np.mean(se2s)
se_std = np.std(se2s)
se2s = np.array([(x - se_mean) / se_std for x in se2s])

print(tdf1s.shape, tdf2s.shape, se1s.shape, se2s.shape)
tdfs = np.concatenate((tdf1s, tdf2s))
ses = np.concatenate((se1s, se2s))
features = np.stack((tdfs, ses), axis=-1)
print(features.shape)
labels = np.concatenate((labels, labels))
      

# splits
feature_train, feature_test, feature_label_train, feature_label_test = train_test_split(features, labels, test_size=0.2)
feature_train, feature_val, feature_label_train, feature_label_val = train_test_split(feature_train, feature_label_train, test_size=0.2)
print((feature_train.shape), (feature_val.shape), (feature_test.shape))
print((feature_label_train.shape), (feature_label_val.shape), (feature_label_test.shape))


model = Sequential()
model.add(Input(shape=(len(features[0]), 2)))
print("64")
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
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
