#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import time
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from pickle import dump, load
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load

#corpus_read
def read_file(filepath):
    with open(filepath) as f:
        str_text = f.read()
    return str_text


text = read_file('dataset.txt')
tokens = text.split(" ")

#sequences
train_len = 3 + 1
text_sequences = []
for i in range(train_len, len(tokens)):
    seq = tokens[i - train_len:i]
    text_sequences.append(seq)

sequences = {}
count = 1
for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1

#tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

# Collecting some information
vocabulary_size = len(tokenizer.word_counts)

n_sequences = np.empty([len(sequences), train_len], dtype='int32')
for i in range(len(sequences)):
    n_sequences[i] = sequences[i]
#split_data
train_inputs = n_sequences[:, :-1]
train_targets = n_sequences[:, -1]

train_targets = to_categorical(train_targets, num_classes=vocabulary_size + 1)
seq_len = train_inputs.shape[1]
train_inputs.shape

#model
def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    opt_adam = optimizers.adam(lr=0.001)
    # we can simply pass 'adam' to optimizer in compile method. Default learning rate 0.001
    # But here we are using adam optimzer from optimizer class to change the LR.
    model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])
    model.summary()
    return model

#train
model = create_model(vocabulary_size + 1, seq_len)
path = './word_pred_Model4.h5'
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit(train_inputs, train_targets, batch_size=128, epochs=1000, verbose=1, callbacks=[checkpoint])
dump(tokenizer, open('tokenizer_Model4', 'wb'))

#Word_Prediction
model = load_model('word_pred_Model4.h5')
tokenizer = load(open('tokenizer_Model4', 'rb'))
seq_len = 3

#show 1 prediction & 1 suggestion #seed.

def gen_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []
    input_text = seed_text
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]

        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += ' ' + pred_word
        output_text.append(pred_word)
    return ' '.join(output_text)


print('\n\n===>Enter --exit to exit from the program')
while True:
    seed_text = input('Enter string: ')
    if seed_text.lower() == '--exit':
        break
    else:
        out = gen_text(model, tokenizer, seq_len=seq_len, seed_text=seed_text, num_gen_words=5)
        print('Output: ' + seed_text + ' ' + out)
