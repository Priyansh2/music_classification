import os,sys,pickle
import string
import re
from collections import defaultdict,Counter
import json
corpus_path=os.getcwd()
file=open("all_lyrics.pkl","rb")
corpus = pickle.load(file)
puncts = list(string.punctuation)
my_delims = "!","…",",",".","-","’"	
my_delims = list(my_delims)
puncts = set(puncts+my_delims)
delimeters = list(puncts)
#regx = re.compile('\W+')
regx = re.compile('[^a-zA-Z]')
result = set(regx.findall(corpus))
result = list(result)
delim_matched_lines=defaultdict(list)
for delim in result:
	for line in corpus.split("\n"):
		if delim in line:
			delim_matched_lines[delim].append(line)

for delim in delim_matched_lines:
	delim_matched_lines[delim]=list(set(delim_matched_lines[delim]))[:5]

##Printing top5 lines belong to every delim				
#print(json.dumps(delim_matched_lines,indent=4))

delimeters = set(delimeters) | set(result)
exclude_delims=set(["-","’",' '])
delimeters = delimeters - exclude_delims
delimeters = list(delimeters)
print(delimeters)
file = open("delim.pkl","wb")
pickle.dump(delimeters,file)
file.close()
for delim in delimeters:
	corpus = corpus.replace(delim,"")
