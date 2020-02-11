import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#reading data from json file
with open("intents.json") as file:
    data = json.load(file)

try: #if the json file is changed  just delete the model.tflearn files and rerun the model
    with open("data.pickle", "rb") as f: #rb = read bytes
        words, labels, training, output = pickle.load(f) #save all var into pickle file
except:
    #creating empty lists
    words = []
    labels = []
    docs_x = [] #pattern
    docs_y = [] #intent

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #sorting and get rid of duplicates
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #one hot encoding for our words
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
    #normalize the words
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

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training,output), f)

#model
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8) #hidden layers 8 neurals each
net = tflearn.fully_connected(net, 8) #hidden layers 8 neurals each

net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #propability for each neuron
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True )
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words=nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s in s_words:
        for i,w in enumerate(words):
            if w == s :
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Start talking with me pal (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower()== "quit":
            break

        answers = model.predict([bag_of_words(inp, words)])
        answers_index= numpy.argmax(answers) #index of the biggest propability answer
        tag = labels[answers_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
