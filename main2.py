import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("ydata-110_examples.text.json") as file:
	data = json.load(file)


words = []
labels = []
docs_x = []
docs_y = []

for question in data["questions"]:
	wrds = nltk.word_tokenize(question["title"])
	words.extend(wrds)
	docs_x.append(wrds)
	docs_y.append(question["question_id"])

	if question["question_id"] not in labels:
		labels.append(question["question_id"])

words = [stemmer.stem(w) for w in words if w != "?"]
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

training = numpy.array(training)
output = numpy.array(output)

with open("data5.pickle", "wb") as f:
	pickle.dump((words, labels, training, output), f)

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=9, show_metric=True)
model.save("model4.tflearn")

def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(w.lower()) for w in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i]=1

	return numpy.array(bag)

def chat():
	print("==================================================")

	while True:
		inp = input("You: ")
		if(inp.lower()=="quit"):
			break

		result = model.predict([bag_of_words(inp, words)])[0]
		result_index = numpy.argmax(result)
		tag = labels[result_index]

		if result[result_index] > 0.4:
			for t in data["questions"]:
				if t["question_id"] == tag:
					for answer in t["answers"]:
						responses = answer["answer_text"]
						if answer["is_best_answer"]==True:
							responses = answer["answer_text"]
							break

			print(responses)
		else:
			print("Not sure, ask something else")

chat()