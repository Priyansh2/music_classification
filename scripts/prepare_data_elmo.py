### DATA preparation script for ELMO ####
import os,dill as pickle
from sklearn.model_selection import train_test_split
from collections import Counter,defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import operator

def data_split(corpus,output_dir):
	Xtrain, Xval = train_test_split(text,test_size=0.1,random_state=4242)
	print("Sentences in training data: ",len(Xtrain))
	print("Sentences in validation data: ",len(Xval))
	write_in_file(output_dir+"/xtrain.txt",Xtrain)
	write_in_file(output_dir+"/xval.txt",Xval)
	#save_model("elmo_data/xtrain.pkl",Xtrain)
	#save_model("elmo_data/xval.pkl",Xval)

def save_model(file_path,model):
	file = open(file_path,"wb")
	pickle.dump(model,file)
	file.close()

def load_model(file_path):
	file = open(file_path,"rb")
	model = pickle.load(file)
	file.close()
	return model

def write_in_file(filename,data):
	print("writing in file...")
	fd = open(filename,"w",encoding="utf-8")
	for sent in data:
		fd.write(sent+"\n")
	fd.close()
	print("Done!!")

datapath="/home/priyansh.agrawal/indic_nlp_library/cm_final_min5.txt"
output_dir="elmo_data_cm"

#datapath="/home/priyansh.agrawal/music_classification/songs_corpus.txt"
#output_dir="elmo_data_songs"
text = open(datapath,"r").read().split("\n")
data_split(text,output_dir)
xtrain= open(output_dir+"/xtrain.txt",'r').read().split("\n")
xval = open(output_dir+"/xval.txt",'r').read().split("\n")
def create_vocab_file(xtrain,output_dir,is_save=False):
	words=[]
	for sent in xtrain:
		sent=sent.strip()
		if sent:
			words.extend(sent.split())
	fd = nltk.FreqDist(words)
	vocab = dict(sorted(fd.items(),key=operator.itemgetter(1),reverse=True))
	n_tokens = sum(vocab.values()) ##Write this number in 'bilm-tf/bin/train_elmo.py' file before training elmo
	print("Total tokens in training data: ",n_tokens)
	special_chars=["</S>","<S>","<UNK>"]
	special_chars.extend(vocab)
	print("Vocabulary Count: ",len(special_chars))
	if is_save:
		write_in_file(output_dir+"/tr_vocab.txt",special_chars)

create_vocab_file(xtrain,output_dir,is_save=True)
#create_vocab_file(xtrain,output_dir)
