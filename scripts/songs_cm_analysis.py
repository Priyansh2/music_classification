#!/usr/bin/env python
# coding: utf-8

import os,sys,dill as pickle,string,re,json,nltk
from litcm import LIT
from three_step_decoding import *
from collections import defaultdict,Counter
from os.path import expanduser
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
home = expanduser("~")
tsd = ThreeStepDecoding('lid_models/hinglish', htrans='nmt_models/rom2hin.pt', etrans='nmt_models/eng2eng.pt')
lit = LIT(labels=['hin', 'eng'], transliteration=False)

def save_model(file_path,model):
	file = open(file_path,"wb")
	pickle.dump(model,file)
	file.close()

def load_model(file_path):
	file = open(file_path,"rb")
	model = pickle.load(file)
	file.close()
	return model

## Code Mixing Index (CMI)
def N(x):
	# Measure Number of token in x which belong to some language
	line=x
	temp=[]
	li_tags={}
	if len(line.split())==1:
		lang = lit.identify(line).split("\\")[1]
		if lang in ("Eng","Hin"):
			if lang=="Eng":
				lang="en"
			else:
				lang="hi"
			temp.append(lang)
	else:
		output = list(tsd.tag_sent(line.decode('utf-8')))
		for w,w_trans,lang in output:
			if lang in ("en","hi"):
				temp.append(lang)
	return temp

def pre_compute_N(genre_data_map):
	data_lang_tags=defaultdict(lambda: defaultdict(lambda: defaultdict()))
	for genre in genre_data_map:
		genre_songs = genre_data_map[genre]
		for song in genre_songs:
			for line in song:
				if len(line.split())==0:
					continue
				data_lang_tags[genre][tuple(song)][line]=N(line)
	save_model("/home/priyansh.agrawal/music_classification/pickled_data/data_lang_tags.pkl",data_lang_tags)

def decide_cmi_threshold(verbose=True):
	cnt=0
	cm_twitter_cmi_scores = load_model("/home/priyansh.agrawal/indic_nlp_library/pickled_data/cm_twitter_cmi_scores.pkl")
	for cmi_score in cm_twitter_cmi_scores:
		if cmi_score==0.0:
			cnt+=1
	avg = float(sum(cm_twitter_cmi_scores))/len(cm_twitter_cmi_scores)
	if verbose:
		print("Total code-mixed tweets: ",len(cm_twitter_cmi_scores))
		print("Number of code-mixed tweets with zero cmi: ",cnt)
		print("Number of code-mixed tweets with non-zero cmi: ",len(cm_twitter_cmi_scores)-cnt)
		print("Percentage of code-mixed tweets with zero cmi: ",float(cnt)/len(cm_twitter_cmi_scores)*100)
		print("Percentage of code-mixed tweets that are actually code-mixed (non-zero cmi): ",float(len(cm_twitter_cmi_scores)-cnt)*100/len(cm_twitter_cmi_scores))
	print("Average cmi of code-mixed tweets: ",avg)
	print("\n")
	return avg

def sanity_checks(genre_data_map,data_lang_tags):
	cnt1=0
	for genre in genre_data_map:
		for song in genre_data_map[genre]:
			if song:
				cnt1+=1
			else:
				print("LULWA!!!")
	cnt2=0
	for genre in data_lang_tags:
		for song in data_lang_tags[genre]:
			if list(song):
				cnt2+=1
			else:
				print("lol!!")
	assert cnt1==cnt2
	cnt=0
	for genre in genre_data_map:
		for song in genre_data_map[genre]:
			for line in song:
				try:
					N_val = data_lang_tags[genre][tuple(song)][line]
				except KeyError:
					cnt+=1
					print(genre,song,[line])
	assert cnt==0
	print("Check Passed!!\n")

def delta(W_max_curr,W_max_prev):
	if W_max_prev==-1:
		return 0
	else:
		if W_max_prev==W_max_curr:
			return 0
		else:
			return 1

def CMI(N_x,lang_tags):
	if not N_x:
		return 0
	else:
		flag=0
		if "en" in N_x and "hi" in N_x:
			W_max = max(N_x["hi"],N_x["en"])
			W_total = sum(N_x.values())
			F =  float(W_total-W_max)/W_total
			flag=1
		elif "en" in N_x or "hi" in N_x:
			F=0
			flag=1
		if flag==1:
			P=0
			W_total = sum(N_x.values())
			try:
				l1 = lang_tags[0]
			except:
				print(lang_tags)
			for i in range(1,len(lang_tags)):
				if lang_tags[i]!=l1:
					l1 = lang_tags[i]
					P+=1
			P=float(P)/W_total
			return 0.5*F + float(5*P)/6

def CMI_song(song,genre):
	sum_=0
	t=0
	for x in range(len(song)):
		line = song[x]
		W_max_prev=-1
		li_tags_curr = data_lang_tags[genre][tuple(song)][line]
		W_max_curr=0
		fd_curr={}
		if li_tags_curr:
			fd_curr = nltk.FreqDist(li_tags_curr)
			W_max_curr = max(fd_curr["hi"],fd_curr["en"])
		if x>0:
			prev_line = song[x-1]
			W_max_prev=0
			li_tags_prev = data_lang_tags[genre][tuple(song)][prev_line]
			if li_tags_prev:
				fd_prev = nltk.FreqDist(li_tags_prev)
				W_max_prev = max(fd_prev["hi"],fd_prev["en"])
		cmi_line = CMI(fd_curr,li_tags_curr)
		if cmi_line>0:
			t+=1
		sum_+=cmi_line+0.5*delta(W_max_curr,W_max_prev)
	sum_+=float(t*5)/6
	try:
		sum_/=len(song)
	except:
		print(genre,[song],len(song))
		sys.exit()
	return sum_

# # Language Identification
def find_cm_songs(genre_data_map,data_lang_tags,th):
	print("Threshold for cmi :",th)
	print("\n")
	cnt_songs=0
	cnt_cm_songs=0
	sum_=0
	cnt=0
	cmi_mixed=0
	cmi_all=0
	CM_genre_map_data=defaultdict(lambda: defaultdict())
	for genre in genre_data_map:
		genre_songs = genre_data_map[genre]
		for song in genre_songs:
			##every song is a list of all utterances (line)
			##computing CMI of every song.
			if song:
				cnt_songs+=1
				cmi = CMI_song(song,genre)
				cmi_all+=cmi
				if cmi>0:
					cmi_mixed+=cmi
					cnt+=1
				if cmi>=th:
					cnt_cm_songs+=1
					CM_genre_map_data[genre][tuple(song)]=cmi
	print("Total songs: ",cnt_songs)
	print("Songs with zero cmi: ",cnt_songs-cnt)
	print("Songs with non-zero cmi: ",cnt)
	cmi_all = float(cmi_all)/cnt_songs
	cmi_mixed=float(cmi_mixed)/cnt
	print("CMI_ALL (%): ",cmi_all*100) ##average over all utterances
	print("CMI_MIXED (%): ",cmi_mixed*100) ##average over the utterances having a non-zero CMI
	print("Percentage of mixing: ",float(cmi_all)*100/cmi_mixed)
	##'CMI-ALL' is a measures to understand how much mixed the corpus is whereas 'CMI-MIXED' is a measure to understand how much mixed all the Code-Mixed utterances are in any corpora.
	print("Songs with cmi_score >= threshold: ",cnt_cm_songs)
	print("Songs with cmi_score < threshold: ",cnt_songs-cnt_cm_songs)
	print("Percentage of songs with cmi_score >= threshold: ",float(cnt_cm_songs)*100/cnt_songs)
	print("Percentage of songs with cmi_score < threshold: ",float(cnt_songs-cnt_cm_songs)*100/cnt_songs)
	save_model("/home/priyansh.agrawal/music_classification/pickled_data/CM_genre_map_data.pkl",CM_genre_map_data)


genre_data_map = load_model("/home/priyansh.agrawal/music_classification/pickled_data/genre_data_map.pkl")
#pre_compute_N(genre_data_map)
data_lang_tags = load_model("/home/priyansh.agrawal/music_classification/pickled_data/data_lang_tags.pkl")
sanity_checks(genre_data_map,data_lang_tags)
avg = decide_cmi_threshold()
th1=avg ##0.21333144905159612
th4=float(3)/7 ##30:70% code-switching
th2=0.5 ##50:50% code-switching
th3=float(2)/3##40:60% code-switching
#ths=[th1,th2,th3,th4]
ths=[float(i)/10 for i in range(1,10)] ##0.3 is better
for th in ths:
	find_cm_songs(genre_data_map,data_lang_tags,th)
	print("\n\n")
#find_cm_songs(genre_data_map,data_lang_tags,0.3)



