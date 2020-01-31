import json
from collections import defaultdict

with open("ydata-110_examples.text.json") as f:
	bi = json.load(f)

newf = []
qa = {}
tag = []
j = 0
for intent in bi["questions"]:
	for _ in enumerate(intent["Question"]):
		j+=1
		if intent["Question"] not in qa:
			qa[intent["Question"]] = intent["Answer"]
			tag.append(intent["ArticleTitle"]+ str(j))
		elif qa[intent["Question"]] == "NULL":
			qa[intent["Question"]] = intent["Answer"]

i = 0
for x,y in qa.items():
	ele = {}
	ele["tag"] = tag[i]
	ele["patterns"] = [x]
	ele["responses"] = [y]
	ele["context_set"] = ""
	newf.append(ele)
	i+=1
 
data = {}
data["intents"] = newf
print(newf[0])

with open("intents2.json", 'w') as file:
	json.dump(data,file,indent=4)