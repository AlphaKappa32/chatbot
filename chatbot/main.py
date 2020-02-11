import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

#reading data from json file
with open("intents.json") as file:
    data = json.load(file)

#creating empty lists
words = []
labels = []
docs_x = [] #pattern
docs_y = [] #intent

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = ntlk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

#sorting and get rid of duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

labels = sorted(labels)

#one hot encoding for our words
training = []
output = []

out_empty = [0 for _ in range(len(classes))]

for x, doc in enumerate(docs_x):
    bag = []
###########
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1) #word exists
        else:
            bag.append(0) #word does not exists

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)
