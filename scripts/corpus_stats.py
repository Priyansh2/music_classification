## incomplete_script
# with script i found that we need a code-mixed word segmenter as many words are found to be joined with each other, which means the combined word is rare because of less presence of it in corpus and this degrades the performance of language model trained using methods like elmo

import os,sys,dill as pickle,operator,re,nltk,numpy as np
from collections import Counter,defaultdict
#data_path=os.getcwd()+"/elmo_scripts/elmo_data_songs"
data_path=os.getcwd()+"/elmo_scripts/elmo_data_cm"
data=[]
data=open(data_path+"/xtrain.txt","r").read().split("\n")+open(data_path+"/xval.txt","r").read().split("\n")
sents=[sent.strip() for sent in data if sent.strip()]
print(len(sents))
def max_char_in_tokens(sents):
	max_char_len=-1
	temp=[]
	for sent in sents:
		for word in sent.split():
			if word!="<EOS>" and word!="<BOS>":
				char_len = len(word.strip())
				#if char_len>4:
				temp.append((word,char_len))
	temp = dict(temp)
	temp = dict(sorted(temp.items(),key=operator.itemgetter(1),reverse=True))
	print(temp)
	sys.exit()
	for sent in sents:
		for word in sent.split():
			if word!="<EOS>" and word!="<BOS>":
				char_len = len(word.strip())
				if char_len>th and max_char_len<char_len:
					max_char_len=char_len
					candidate_word=word
	return max_char_len,candidate_word
print(max_char_in_tokens(sents))
