##Require python2.7
###NOTE :- Upon analysis of LI tags I found that the lyrics dataset contains
# following different tags :- 'en' (for english), 'univ' (for universal),
# 'acro' (for acronymns), 'ne' (for named entity), 'hi' (for hindi)
# This means two languages i.e, Hindi and English are present in song lyrics.
# CMI (Code-mixed index) takes two language set (L) i.e, hi and en.
# 'acro','ne','univ' are all non-language tags given to tokens
import os,pickle,sys,re
from collections import defaultdict,Counter
try:
	from os.path import expanduser
	home = expanduser("~")
except ImportError:
	from pathlib import Path
	home = str(Path.home())
csnli_path=os.path.join(home,"csnli")
from three_step_decoding import *
code_source=os.path.join(csnli_path,"codes/python-itunes")
lyrics_data_path=os.path.join(code_source,"data/lyrics")
genre_dirs = sorted(os.listdir(lyrics_data_path))

def LI():
	li_tags=[]
	all_lyrics=[] ##list of tuple where first entry of tuple is
	# song_lyrics_file_path and second is list containing each line of its
	# corresponding lyrics (preserving the original format)
	corpus='' ##single text file like corpus which contains each non-empty line of all song
	# lyrics of dataset
	tsd = ThreeStepDecoding('lid_models/hinglish', htrans='nmt_models/rom2hin.pt', etrans='nmt_models/eng2eng.pt')
	for genre in genre_dirs:
		for file in sorted(os.listdir(os.path.join(lyrics_data_path,genre))):
			file_path = os.path.join(os.path.join(lyrics_data_path,genre),file)
			#print(file_path)
			lyrics = open(file_path,"r").readlines()
			#print(lyrics)
			trans_lyrics=[]
			prev=False
			prev_str=None
			for line in lyrics:
				if line.split("\n")[0].strip():
					input_=line.split("\n")[0].lower()
					corpus+=input_+"\n"
					if len(input_.split())==1:
						if prev:
							prev_str+=" "+input_
						else:
							prev=True
							prev_str=input_
						continue
					if prev_str!=None:
						l=len(prev_str.split())
						input_=prev_str+" "+input_
					word_str=''
					try:
						output_=list(tsd.tag_sent(input_))
					except Exception as e:
						print("Error"," , ",file_path," , ",lyrics)
						print([input_])
						sys.exit()
					assert len(input_.split())==len(output_)
					for w,trans_w,lang in output_[:-1]:
						li_tags.append(lang)
						word_str+=w+"$$$$$"+trans_w+"$$$$$"+lang+" "
					word_str+=output_[-1][0]+"$$$$$"+output_[-1][1]+"$$$$$"+output_[-1][2]
					li_tags.append(output_[-1][2])
					rem_str = word_str
					if prev:
						for i in range(l):
							trans_lyrics.append(word_str.split()[i]+"\n")
						rem_str = " ".join(x for x in word_str.split()[l:])
						prev=False
						prev_str=None
					trans_lyrics.append(rem_str+"\n")
				else:
					trans_lyrics.append("\n")
			#break
			#print(trans_lyrics)
			all_lyrics.append((file_path,trans_lyrics))
		#break
	return corpus,li_tags,all_lyrics

def save_model(file_path,model):
	file = open(file_path,"wb")
	pickle.dump(model,file)
	file.close()

def load_model(file_path):
	file = open(file_path,"rb")
	model = pickle.load(file)
	file.close()
	return model

#corpus,li_tags,all_lyrics = LI()
#print("Unique LI tags: ",set(li_tags))
#save_model(os.path.join(csnli_path,"trans_lyrics.pkl"),all_lyrics)
#save_model(os.path.join(csnli_path,"all_lyrics.pkl"),corpus)
all_lyrics = load_model(os.path.join(csnli_path,"trans_lyrics.pkl"))
blacklist=[
"/home/priyansh.agrawal/csnli/codes/python-itunes/data/lyrics/dance/2018_Genius_Pyar Le Pyar De.txt",
"/home/priyansh.agrawal/csnli/codes/python-itunes/data/lyrics/happy/1976_Sajjo Rani_Saiya Ke Gaon Me.txt",
"/home/priyansh.agrawal/csnli/codes/python-itunes/data/lyrics/happy/2018_Baa Baaa Black Sheep_Galla Goriyan.txt",
"/home/priyansh.agrawal/csnli/codes/python-itunes/data/lyrics/love/2016_Shivaay_Darkhaast.txt"]
delims = load_model(os.path.join(csnli_path,"delim.pkl"))
print("\n\nDelimiters are: ",delims)
transliterated_data_path = os.path.join(code_source,"data/transliterated/lyrics")
if not os.path.exists(transliterated_data_path):
	os.makedirs(transliterated_data_path)
def write_lyrics_to_file(all_lyrics):
	#global tagged_corpus
	global csnli_path
	tagged_corpus=''
	for original_file_path,trans_lyrics in all_lyrics:
		if original_file_path in blacklist:
			continue
		genre = original_file_path.split("/")[-2]
		genre_dir = os.path.join(transliterated_data_path,genre)
		if not os.path.exists(genre_dir):
			os.makedirs(genre_dir)
		output_file_path = os.path.join(genre_dir,original_file_path.split("/")[-1])
		#print(original_file_path)
		fd= open(output_file_path,"w")
		#print(trans_lyrics)
		for line in trans_lyrics:
			str1=''
			if line.split("\n")[0].strip():
				'''cnt=0
				for x in line:
					if x=="/":
						cnt+=1
				test = cnt/2
				try:
					assert 2*int(test)==cnt
				except AssertionError:
					print("error_1",original_file_path,line,trans_lyrics)
					sys.exit()'''
				tagged_corpus+=line
				words = line.split("$$$$$")
				#words = line.split()
				temp=[]
				for x in range(len(words)):
					if len(words[x].split())==2:
						temp+=words[x].split()
					else:
						temp+=[words[x]]
				words=temp
				#print(words)
				for x in range(1,len(words),3):
					if not words[x].strip():
						try:
							assert 1==2
						except AssertionError:
							print("error_2",original_file_path,line,trans_lyrics)
							sys.exit()
					str1+=words[x]+" "
				str1.rstrip()
				for delim in delims:
					if delim in str1:
						str1=str1.replace(delim,"")

				str1=str1.strip()
			str1+="\n"
			fd.write(str1)
		fd.close()
		#print("done.....")
	save_model(os.path.join(csnli_path,"all_tagged_lyrics.pkl"),tagged_corpus)
#write_lyrics_to_file(all_lyrics)

def verify_content(original_data,transliterated_data):
	global blacklist
	for genre in genre_dirs:
		dir1 = sorted(os.listdir(os.path.join(original_data,genre)))
		dir2 =  sorted(os.listdir(os.path.join(transliterated_data,genre)))
		temp=[]
		bkl_files=[file.split("/")[-1] for file in blacklist]
		#for file in blacklist:
			#print(file.split("/")[-1])
		for file in dir1:
			if file not in bkl_files and file not in temp:
				temp.append(file)
		dir1=temp
		try:
			assert len(dir1)==len(dir2)
		except AssertionError:
			print("Genre: ",genre,"No. of orig. files: ",len(dir1),"No. of trans. files: ",len(dir2))
			if len(dir1)>len(dir2):
				for file in dir1:
					if file not in dir2:
						print(file)
			else:
				for file in dir2:
					if file not in dir1:
						print(file)
			sys.exit()
		for orig_file,trans_file in zip(dir1,dir2):
			assert orig_file==trans_file
			file1=os.path.join(os.path.join(original_data,genre),orig_file)
			file2 = os.path.join(os.path.join(transliterated_data,genre),trans_file)
			orig_lyrc = open(file1).readlines()
			trans_lyrc = open(file2).readlines()
			#assert len(orig_lyrc)==len(trans_lyrc)
			for line1,line2 in zip(orig_lyrc,trans_lyrc):
				if line1.strip() and line2.strip():
					for delim in delims:
						if delim in line1:
							line1=line1.replace(delim,"")
					try:
						assert len(line1.split())==len(line2.split())
					except AssertionError:
						for x,y in all_lyrics:
							if x==os.path.join(os.path.join(lyrics_data_path,genre),trans_file):
								print(y)
								break
						print(orig_file," , ",trans_file," , ",genre," , ",line1," , ",line2," , ",orig_lyrc,"\n\n",trans_lyrc)
						#for x,y in all_lyrics:
							#if x.split("/")[-1]=="1955_Jhanak Jhanak Payal Baaje_Jo Tum Todo Piya.txt":
								#print("\n\n",y)
						sys.exit()
#print(lyrics_data_path,transliterated_data_path)
verify_content(lyrics_data_path,transliterated_data_path)
