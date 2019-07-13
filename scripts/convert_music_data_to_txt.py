import os,dill as pickle,sys
def write_in_file(filename,data):
	print("writing in file...")
	fd = open(filename,"w",encoding="utf-8")
	for sent in data:
		fd.write(sent+"\n")
	fd.close()
	print("Done!!")

def save_model(file_path,model):
	file = open(file_path,"wb")
	pickle.dump(model,file)
	file.close()

def load_model(file_path):
	file = open(file_path,"rb")
	model = pickle.load(file)
	file.close()
	return model


datapath="pickled_data/genre_data_map.pkl"
genre_data_map = load_model(datapath)
all_songs_lyrics=[]
for genre in genre_data_map:
	for song in genre_data_map[genre]:
		str_=''
		for line in song:
			if line.strip():
				str_+="<BOS> "+line.strip()+" <EOS> "
		all_songs_lyrics.append(str_.strip())
print("Songs in data: ",len(all_songs_lyrics))
write_in_file("songs_corpus.txt",all_songs_lyrics)
