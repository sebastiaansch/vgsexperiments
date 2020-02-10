import pandas as pd
import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from os import path
from pydub import AudioSegment

def tokenize(path):
    text = pd.read_csv(path, sep="\t")
    text = text.iloc[:, 1]
    tokenizer = RegexpTokenizer(r'\w+')
    wordlist = []
    for sentence in text:
        sentence = sentence.lower()
        words = tokenizer.tokenize(sentence)
        wordlist.extend(words)
    wordlist = [x.lower() for x in wordlist]
    sr = pd.Series(wordlist)
    worddictionary = dict(sr.value_counts())
    return wordlist, worddictionary

def returnnouns(x):
    nouns = [words for idx,words in enumerate(x) if x[idx][1] == "NN"]
    return nouns

def create_noun_set(wordlist,data_file):
    mfcc = [node for node in iterate_data(data_file)]
    mfccset = []
    for word in mfcc:
        raw_word = word._v_name.replace('flickr_', '')
        for wordlistword in wordlist:
            if raw_word == wordlistword:
                mfccset.append(word)
    return mfccset

def iterate_data(h5_file):
    for x in h5_file.root:
        yield x

def dicttodf(dictionary):
    dfwordlist = pd.DataFrame.from_dict(dictionary,orient="index")
    dfwordlist.reset_index(level=0, inplace=True)
    keys = dfwordlist.iloc[:,0]
    values = dfwordlist.iloc[:,1]
    dfwordlist.columns = ["words","counts"]
    return dfwordlist

def MP3toWAV(audio,fromdirectory,todirectory):
    sound = AudioSegment.from_mp3(os.path.join(fromdirectory,audio))
    sound.export(os.path.join(todirectory,audio[:-4] + ".wav"), format="wav")