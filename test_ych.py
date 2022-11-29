# Larger LSTM Network to Generate Text for Alice in Wonderland
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, encoding="utf8").read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=100, batch_size=64, callbacks=callbacks_list)

# ['\n', '\r', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']',
# '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
# 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xbb', '\xbf', '\xef']


fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56],
		[2.8197, 2.4243, 2.2148, 2.0802, 1.9853, 1.9089, 1.8458, 1.7958, 1.7460, 1.7158, 1.6864, 1.6515, 1.6187, 1.5908, 1.5562, 1.5284, 1.5484, 1.5072, 1.4906,
		 1.4673, 1.4493, 1.4324, 1.4185, 1.4056, 1.3905, 1.3772, 1.33665, 1.13550, 1.3425, 1.3360, 1.3247, 1.3160, 1.3148, 1.2986, 1.2937, 1.2839, 1.2809, 1.2750,
		 1.2902, 1.2667, 1.2591, 1.2752, 1.2495, 1.2435, 1.2426, 1.2342, 1.2340, 1.2313, 1.2285, 1.2248, 1.2347, 1.2343, 1.2293, 1.2272, 1.2231, 1.2202]);





