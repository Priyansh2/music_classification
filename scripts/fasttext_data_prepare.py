import os,sys,numpy as np,dill as pickle
from random import shuffle
from collections import defaultdict
def prepare_data_for_fasttext(X,Y,filename):
	## Prepare raw data for fasttext model training
	data=''
	fd = open(filename,"w",encoding='utf-8')
	for i in range(len(X)):
		sent = X[i]
		label = Y[i]
		data+="__label__"+str(label)+" "+sent+"\n"
	fd.write(data)
	fd.close()


def save_model(file_path,model):
	file = open(file_path,"wb")
	pickle.dump(model,file,protocol=2)
	file.close()

def load_model(file_path):
	file = open(file_path,"rb")
	model = pickle.load(file)
	file.close()
	return model

def train_dev_split(x,do_shuffle=True):
	if do_shuffle:
		for label in x:
			shuffle(x[label])
	xtrain=defaultdict(list)
	xval=defaultdict(list)
	for label in x:
		i = int(len(x[label])*9/10)
		xtrain[label]=x[label][:i]
		xval[label]=x[label][i:]
	return xtrain,xval
def decompose_x(x):
	X,Y=[],[]
	for label in x:
		for text in x[label]:
			X.append(text)
			Y.append(label)
	assert len(X)==len(Y)
	return X,Y

def extract_data(genre_data_map):
	temp=defaultdict(list)
	for genre in genre_data_map:
		for song in genre_data_map[genre]:
			str_=''
			for line in song:
				if line.strip():
					str_+="<BOS> "+line.strip()+" <EOS> "
			temp[genre].append(str_.strip())
	return temp

datapath="pickled_data/genre_data_map.pkl"
genre_data_map = load_model(datapath)
xtrain,xval=train_dev_split(extract_data(genre_data_map))
Xtrain,Ytrain=decompose_x(xtrain)
Xval,Yval = decompose_x(xval)
#prepare_data_for_fasttext(Xtrain,Ytrain,"ft_data/music/songs_ft.train")
#prepare_data_for_fasttext(Xval,Yval,"ft_data/music/songs_ft.dev")
