import itertools
import os
import sys
import random
import math
import dill as pickle
import csv
import json
import multiprocessing
import numpy as np
from collections import Counter
from collections import defaultdict
from pathlib import Path
from random import shuffle
import re
import string


def save_model(file_path,model):
	file = open(file_path,"wb")
	pickle.dump(model,file,protocol=2)
	file.close()

def load_model(file_path):
	file = open(file_path,"rb")
	model = pickle.load(file)
	file.close()
	return model

def write_in_file(filename,data):
	##data is list of strings
	##filename is file_path
	print("Writing content in file....")
	fd = open(filename,"w",encoding="utf-8")
	for sent in data:
		fd.write(sent+"\n")
	fd.close()
	print("Done!!")

def tokenise(line):
    delims = load_model(os.path.join(csnli_path,"delim.pkl"))
    line = line.lower()
    for delim in delims:
        line = line.replace(delim,"")
    return line

def read_genre_data(genre,lyrics_data_path,tracker,is_tokenise=True):
    corpus=defaultdict(list)
    for file in sorted(os.listdir(os.path.join(lyrics_data_path,genre))):
        doc=[]
        song_name = file.split("_")[-1].split(".txt")[0].strip()
        movie_name = file.split("_")[1].strip()
        movie_year = file.split("_")[0].strip()
        file_path = os.path.join(os.path.join(lyrics_data_path,genre),file)
        if os.stat(file_path).st_size==0:
            #print(genre,file_path)
            continue
        lyrics = open(file_path,"r").readlines()
        for line in lyrics:
            if line.split("\n")[0].strip():
                input_=line.split("\n")[0]
                if is_tokenise:
                    input_=tokenise(input_).strip()
                if len(input_.split())>0:
                    doc.append(input_)


        if doc not in corpus["lyrics"] and doc not in tracker:
            tracker.append(doc)
            corpus["song_names"].append(song_name)
            corpus["movie_names"].append(movie_name)
            corpus["movie_years"].append(movie_year)
            corpus["lyrics"].append(doc)

    return corpus,tracker

cwd = os.getcwd()
home=str(Path.home())
csnli_path=os.path.join(home,"csnli")
code_source=os.path.join(cwd,"python_itunes")
lyrics_data_path=os.path.join(code_source,"data/lyrics")
genre_dirs = sorted(os.listdir(lyrics_data_path))

## data is dictionary where key is 'genre' mapped to list of lists (Each list correpsonds to 'song lyrics')
data=[]
tracker=[]
for genre in genre_dirs:
    corpus,tracker=read_genre_data(genre,lyrics_data_path,tracker)
    assert len(corpus["song_names"])==len(corpus["movie_names"])==len(corpus["movie_years"])==len(corpus["lyrics"])
    for i in range(len(corpus["song_names"])):
    	song_name = corpus["song_names"][i]
    	movie_name = corpus["movie_names"][i]
    	movie_year = corpus["movie_years"][i]
    	lyrics = corpus["lyrics"][i]
    	container={"song_name":song_name,
            "movie_name":movie_name,
            "movie_year":movie_year,
            "song_genre":genre,
            "lyrics":lyrics}
    	data.append(container)
print(len(data))
save_model("spotify_data.pkl",data)
