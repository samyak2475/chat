import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

nltk.download('punkt_tab', quiet=True)

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
import pickle
import random 
import json

with open("intents.json") as file:
    data = json.load(file)
    
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        
words = [stemmer.stem(w.lower())for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
        
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

training = np.array(training)
output = np.array(output)

model = Sequential([layers.Dense(8, activation='relu', input_shape=(len(training[0]),)), layers.Dense(8, activation='relu'), layers.Dense(len(output[0]), activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(training, output, epochs=100, batch_size=8, verbose=0)
model.save("model.keras")

# Evaluate and show accuracy
loss, accuracy = model.evaluate(training, output, verbose=0)
print(f"Training Accuracy: {accuracy*100:.2f}%")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s in s_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
        
    return np.array(bag)

def chat():
    print("Start talking with the bot")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        bow = bag_of_words(inp, words)
        bow = bow.reshape(1, -1)  # Ensure correct shape for model
        results = model.predict(bow, verbose=0)
        results_index = np.argmax(results)
        tag = labels[results_index]
        #print(tag)
        #print(f"Results: {results}")

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))

chat()

# import nltk
# from nltk.stem.lancaster import LancasterStemmer
# stemmer = LancasterStemmer()

# import numpy as np
# import tflearn
# import tensorflow as tf
# import random 
# import json

# with open("intents.json") as file:
#     data = json.load(file)

# words = []
# labels = []
# docs_x = []
# docs_y = []

# for intent in data["intents"]:
#     for pattern in intent["patterns"]:
#         wrds = nltk.word_tokenize(pattern)
#         words.extend(wrds)
#         docs_x.append(wrds)
#         docs_y.append(intent["tag"])

#     if intent["tag"] not in labels:
#         labels.append(intent["tag"])
        
# words = [stemmer.stem(w.lower())for w in words if w != "?"]
# words = sorted(list(set(words))) #set() is used to remove duplicates from the list

# labels = sorted(labels)

# training = []
# output = []

# out_empty = [0 for _ in range(len(labels))]

# for x, doc in enumerate(docs_x):
#     bag = []

#     wrds = [stemmer.stem(w) for w in doc]

#     for w in words:
#         if w in wrds:
#             bag.append(1)
#         else:
#             bag.append(0)
        
#     output_row = out_empty[:]
#     output_row[labels.index(docs_y[x])] = 1

#     training.append(bag)
#     output.append(output_row)

# training = np.array(training)
# output = np.array(output)

# tf.reset_default_graph()

# net = tflearn.input_data(shape=[None, len(training[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# net = tflearn.regression(net)

# model = tflearn.DNN(net)

# model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
# model.save("model.tflearn")
