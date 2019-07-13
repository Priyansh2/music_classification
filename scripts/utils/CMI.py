##This script measures Das.A CMI (code-mixed index) of any LI tagged corpus

import os,sys
from collections import defaultdict,Counter
import pickle
try:
	from os.path import expanduser
	home = expanduser("~")
except ImportError:
	from pathlib import Path
	home = str(Path.home())

csnli_path=os.path.join(home,"csnli")


def save_model(file_path,model):
	file = open(file_path,"wb")
	pickle.dump(model,file)	
	file.close()

def load_model(file_path):
	file = open(file_path,"rb")
	model = pickle.load(file)
	file.close()
	return model

corpus = load_model(os.path.join(csnli_path,"all_tagged_lyrics.pkl"))


def N_P_Wli(x):
	##Computes N, P and max(W_li) of any utterance x which is each line of lyrics in our case
	line = x
	words = line.split("$$$$$")
	temp=[]
	for x in range(len(words)):
		if len(words[x].split())==2:
			temp+=words[x].split()
		else:
			temp+=[words[x]]
	words=temp
	cnt=0
	s=0
	lang_seq=[]
	for x in range(2,len(words),3):
		if words[x] in ("en","hi"):
			cnt+=1
			if words[x]=="en":
				lang_seq.append(1)
			elif words[x]=="hi":
				lang_seq.append(0)
	if lang_seq:
		for x in range(1,len(lang_seq)):
			if lang_seq[x]!=lang_seq[x-1]:
				s+=1
	lang_words = Counter(lang_seq)
	dominant_lang_words=max(lang_words[0],lang_words[1])
	return cnt,s,dominant_lang_words

##Utterance level Mixing
def CMI_x(x):
	cnt,s,dominant_lang_words = N_P_Wli(x)	
	return (cnt+s-dominant_lang_words)/(2*cnt)

def U_dash(corpus):
	cnt=0
	for x in corpus:
		if CMI_x(x)>0:
			cnt+=1
			u_dash.append(x)
	return cnt			


def L(i,corpus):
	##Computes max(W_li) for utterance whose id is i
	x = corpus[i]
	_,_,W_li = N_P_Wli(x)
	return W_li

def delta(i,corpus):
	##Computes delta function
	if i==0 or L(i,corpus)==L(i-1,corpus):
		return 0
	else:
		return 1	

## Corpus level mixing 
def CMI(corpus):
	##corpus should be list containing all utterances of every text/doc.
	## This function computes CMI of whole corpus
	u = len(corpus)
	u_dash = U_dash(corpus)
	ans=0
	for i in range(u):
		x = corpus[i]
		cmi = CMI_x(x)
		delta_x = delta(i,corpus)
		cmi = cmi + delta_x/2	
		ans+=cmi 
	ans+=5*u_dash/6
	ans/=u
	return ans	

print("CMI: ",CMI(corpus.split("\n")))




