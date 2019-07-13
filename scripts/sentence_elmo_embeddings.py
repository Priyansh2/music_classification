###Activate "flair" python env before running the script and put 'elmo_models' folder in '/scratch/$USER/' folder by running this command :-
#'mkdir -p /scratch/$USER/elmo_models && rsync -avzP ada:/share1/$USER/word_embeddings/elmo_models/music /scratch/$USER/elmo_models'
import dill as pickle
import os,numpy as np,sys
from sklearn.decomposition import PCA
from typing import List
import math,nltk,operator
from collections import Counter,defaultdict
from itertools import islice

# an embedding word with associated vector
class Word:
	def __init__(self, text, vector):
		self.text = text
		self.vector = vector

	def __str__(self):
		return self.text + ' : ' + str(self.vector)

	def __repr__(self):
		return self.__str__()


# a sentence, a list of words
class Sentence:
	def __init__(self, word_list):
		self.word_list = word_list

	# return the length of a sentence
	def len(self) -> int:
		return len(self.word_list)

	def __str__(self):
		word_str_list = [word.text for word in self.word_list]
		return ' '.join(word_str_list)

	def __repr__(self):
		return self.__str__()


# todo: get a proper word frequency for a word in a document set
# or perhaps just a typical frequency for a word from Google's n-grams
def get_word_frequency(word_text,word_freq):
	if word_freq[word_text]==0:
		return word_freq["<UNK>"] # set to a low occurring frequency - probably not unrealistic for most words, improves vector values
	else:
		return word_freq[word_text]

# A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# Sanjeev Arora, Yingyu Liang, Tengyu Ma
# Princeton University
# convert a list of sentence with word2vec items into a set of sentence vectors
def sentence_to_vec(sentence_list,embedding_size,output_dir,a=1e-3):
	word_freq = load_model(output_dir+"/elmo_prediction_vocab.pkl")
	sentence_set = []
	for sentence in sentence_list:
		vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
		sentence_length = sentence.len()
		for word in sentence.word_list:
			a_value = a / (a + get_word_frequency(word.text,word_freq))  # smooth inverse frequency, SIF
			vs = np.add(vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector
		vs = np.divide(vs, sentence_length)  # weighted average
		sentence_set.append(vs)  # add to our existing re-calculated set of sentences
	# calculate PCA of this sentence set
	pca = PCA()
	pca.fit(np.array(sentence_set))
	u = pca.components_[0]  # the PCA vector
	u = np.multiply(u, np.transpose(u))  # u x uT
	# pad the vector?  (occurs if we have less sentences than embeddings_size)
	if len(u) < embedding_size:
		for i in range(embedding_size - len(u)):
			u = np.append(u, 0)  # add needed extension for multiplication below
	# resulting sentence vectors, vs = vs -u x uT x vs
	sentence_vecs = []
	for vs in sentence_set:
		sub = np.multiply(u,vs)
		sentence_vecs.append(np.subtract(vs, sub))
	return sentence_vecs

def take(n, iterable):
	"Return first n items of the iterable as a list"
	return list(islice(iterable, n))

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
	print("Writing in a file....")
	fd = open(filename,"w",encoding="utf-8")
	for sent in data:
		fd.write(sent+"\n")
	fd.close()
	print("Done!!")

def create_vocab_file(data,output_dir,do_save=False,th=5):
	words=[]
	for sent in data:
		sent=sent.strip()
		if sent:
			words.extend(sent.split())
	fd = nltk.FreqDist(words)
	filtered_vocab={}
	sum_=0
	for word,freq in fd.items():
		if freq>th:
			filtered_vocab[word]=freq
		else:
			sum_+=1
	filtered_vocab["<UNK>"]=sum_
	filtered_vocab = dict(sorted(filtered_vocab.items(),key=operator.itemgetter(1),reverse=True))
	vocab = defaultdict(int,filtered_vocab)
	print("Vocabulary Count: ",len(vocab))
	if do_save:
		print("Saving vocabulary...")
		save_model(output_dir+"/elmo_prediction_vocab.pkl",vocab)

def get_labels(sentences,output_dir,mode="music",do_save=False):
	data={}
	if mode=="music":
		datapath="/home/priyansh.agrawal/music_classification/pickled_data/genre_data_map.pkl"
		genre_data_map = load_model(datapath)
		for genre in genre_data_map:
			for song in genre_data_map[genre]:
				if not song:
					print("lol")
					continue
				str_=''
				for line in song:
					if line.strip():
						str_+="<BOS> "+line.strip()+" <EOS> "
				data[str_.strip()]=genre
	labels=[]
	for sentence in sentences:
		sentence=sentence.strip()
		if sentence:
			if sentence in data:
				labels.append(data[sentence])
			else:
				print("lol")
	if do_save:
		print("Saving Labels in a text file...")
		write_in_file(output_dir+"/Largevis_data_labels.txt",labels)

def gen_sent_vec(model_path,embedding_size,data,output_dir,test_on_sample=True,do_save=True):
	options_file=model_path+"/options.json"
	weight_file=model_path+"/weights.hdf5"
	sentence_vector_lookup = dict()
	if test_on_sample:
		sample_test_data=data[:5]
		sentences=[sent.strip().split() for sent in sample_test_data if sent.strip()]
	else:
		sentences = [sent.strip().split() for sent in data if sent.strip()]
	if do_save:
		from allennlp.modules.elmo import Elmo, batch_to_ids
		elmo = Elmo(options_file, weight_file, 1, dropout=0)
		l,batches,cnt=0,[],1
		for r in range(10,len(sentences),10):
			batches.append(sentences[l:r])
			l=r
		batches.append(sentences[l:])
		print("No. of batches: ",len(batches))
		for sentences in batches:
			character_ids = batch_to_ids(sentences)
			embeddings = elmo(character_ids)["elmo_representations"][0]
			batch_sentence_list=[]
			for i in range(len(sentences)):
				word_list=[]
				sent_len = len(sentences[i])
				word_embeddings = embeddings[i][...].detach().numpy()
				for j in range(sent_len):
					word = sentences[i][j]
					word_vec = word_embeddings[j]
					word_list.append(Word(word,word_vec))
				if len(word_list)==0:
					raise Exception("Sentence has empty word_list!!\n")
				batch_sentence_list.append(Sentence(word_list))
			save_model("/scratch/priyansh.agrawal/"+output_dir+"/batches/sentence_list_"+str(cnt)+".pkl",batch_sentence_list)
			cnt+=1
	else:
		batches_dir = "/scratch/priyansh.agrawal/"+output_dir+"/batches"
		sentence_list=[]
		num_batches = len(os.listdir(batches_dir))
		for i in range(1,num_batches+1):
			filename = os.path.join(batches_dir,"sentence_list_"+str(i)+".pkl")
			batch_sentence_list = load_model(filename)
			sentence_list.extend(batch_sentence_list)
		sentence_vectors = sentence_to_vec(sentence_list, embedding_size,output_dir)
		if len(sentence_vectors)!=len(sentence_list):
			raise Exception("Number of sentence_vectors is not equal to number of sentences!!\n")
		for i in range(len(sentence_vectors)):
			sentence_vector_lookup[sentence_list[i].__str__()] = sentence_vectors[i]
		save_model(output_dir+"/elmo_sentence_vector_lookup.pkl",sentence_vector_lookup)

def prepare_data_for_Largevis(output_dir,embedding_size):
	print("\nPreparing data for Largevis...")
	sentence_vector_lookup = load_model(output_dir+"/elmo_sentence_vector_lookup.pkl")
	n_vecs = len(sentence_vector_lookup)
	vec_dim = embedding_size
	fd = open(output_dir+"/elmo.vec","w")
	fd.write(str(n_vecs)+" "+str(vec_dim)+"\n")
	for sent_text,sent_vec in sentence_vector_lookup.items():
		#print("TEXT: ",sent_text)
		#print("VECTOR: ",sent_vec,"\n\n")
		str_=''
		for element in sent_vec.tolist():
			str_+=str(element)+" "
		fd.write(str_.strip()+"\n")
	fd.close()
	print("Done!!\n")

embedding_size=1024
#output_dir="elmo_data_cm"
#model_path="/scratch/priyansh.agrawal/elmo_models/cm_twitter"
output_dir="elmo_data_songs"
model_path="/scratch/priyansh.agrawal/elmo_models/music"
xtrain= open(output_dir+"/xtrain.txt",'r').read().split("\n")
xval = open(output_dir+"/xval.txt",'r').read().split("\n")
data=xtrain+xval
#create_vocab_file(data,output_dir,do_save=True)
#get_labels(data,output_dir,do_save=True)
#gen_sent_vec(model_path,embedding_size,data,output_dir,test_on_sample=False)
gen_sent_vec(model_path,embedding_size,data,output_dir,test_on_sample=False,do_save=False)
prepare_data_for_Largevis(output_dir,embedding_size)




