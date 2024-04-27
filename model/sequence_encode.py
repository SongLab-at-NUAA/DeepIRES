import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tqdm as tqdm
import numpy as np
import  pandas as pd
from functools import reduce
def load_text_file(file_text):
    with open(file_text) as f:
        lines = f.readlines()
        my_data = [line.strip().upper() for line in lines[1::2]]
        return my_data


def read_test_file(filename):
    text_file = open(filename)
    lines = text_file.readlines()
    m = len(lines) // 5
    my_data = []
    for i in range(m):
        text = lines[i * 5 + 1].strip() + lines[i * 5 + 2].strip() + \
               lines[i * 5 + 3].strip() + lines[i * 5 + 4].strip()
        my_data.append(text.upper())

    return my_data

def get_tokenizer_onehot():
    f = ['A','C','G','T']
    res = []
    for i in f:
        res.append(i)
    NB_WORDS = 5
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer

def sentence2char(str_set):
    char_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr)):
            if('N' in sr[i]):
                tmp.append('null')
            else:
                tmp.append(sr[i])
        char_seq.append(' '.join(tmp))
    return char_seq


def char2num(charseq, tokenizer, MAX_LEN):
   # print(charseq)
    sequences = tokenizer.texts_to_sequences(charseq)
    #print(sequences)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq


def sentence2num_onehot(str_set, tokenizer, MAX_LEN):
    #print(str_set)
    charseq = sentence2char(str_set)
    #print(charseq)
    numseq = char2num(charseq, tokenizer, MAX_LEN)
    return numseq


def get_data_onehot(enhancers,maxlen):
    tokenizer = get_tokenizer_onehot()
    MAX_LEN = maxlen
    X_en = sentence2num_onehot(enhancers, tokenizer, MAX_LEN)


    return X_en

def read_str_file(filename):
    text_file = open(filename)
    lines = text_file.readlines()
    m = len(lines) // 3
    my_data = []
    for i in range(m):
        text = lines[i * 3 + 2][:174].strip()
        my_data.append(text)

    return my_data











