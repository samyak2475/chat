import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random 
import json

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
doc = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        doc.append(pattern)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])