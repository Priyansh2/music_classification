###Activate "flair" python env before running the script and put 'elmo_models' folder in '/scratch/$USER/' folder

import dill as pickle
import os,numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

model_path="/scratch/priyansh.agrawal/elmo_models/cm_twitter"
options_file=model_path+"/options.json"
weight_file=model_path+"/weights.hdf5"
elmo = Elmo(options_file, weight_file, 1, dropout=0)
#output_dir="elmo_data_cm"
output_dir="elmo_data_songs"
dataset_file=output_dir+"/xval.txt"
sample_test_data=open(dataset_file,"r").read().split("\n")[:5]
sentences=[sent.split() for sent in sample_test_data]
character_ids = batch_to_ids(sentences)
embeddings = elmo(character_ids)
for sent in sentences:
	print(len(sent))
	#print(sent,len(sent),"\n\n")
print("Batch size: ",len(sentences))
print("Max seq length: ",max([len(sent) for sent in sentences]))
vec=embeddings["elmo_representations"]
assert len(vec)==1
elmo_vec = vec[0] ##becoz length is 1
print("Shape of elmo vector: ",elmo_vec.shape) ##corresponding to (batch_size, max_seq_len, elmo_embedding_dim).
## Batch size here is the number of sentences for which we want embeddings. (here 5)
##The sentence length is padded and becomes equal to "max_seq_len" which is 40 in this case.
#Hence each sentence is made up of 40 tokens and corresponding to each token there is 1024 dimensional vector.
#It should be noted that for sentences whose number of tokens is less than "max_seq_len" would have zero entries in vector corresponding to word at index greater than its length. see below for 'first_sentence' for example.
#NOTE:- I am using SIF weighting scheme to make final sentence vector. One can make use of any pooling strategy to make sentence vector given its token embeddings.
#One can also train for sentence embeddings with its token embeddings as input in a downstream task.
print("ELMO vectors for all tokens of all sentences: \n\n",elmo_vec)
v1 = elmo_vec[0][...].detach().numpy()
print("ELMO vectors for all tokens of first sentence: ",v1.shape,"\n\n",v1)
v1_list=v1.tolist()
length_first_sentence = len(sentences[0])
print("\n\n",v1_list[length_first_sentence-1])
print("\n\n",v1_list[length_first_sentence])
print("\n\n",v1_list[length_first_sentence+1])


