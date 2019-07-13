#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastText
import itertools
import os
import sys
import random
import math
import numpy as np
import dill as pickle
import csv
import json
import multiprocessing
from collections import Counter
from collections import defaultdict
from pathlib import Path
from random import shuffle
import subprocess
import re
import string
#corpus_file="cm_twiiter_ft.train"
#output_dir="cm_twitter"

corpus_file="songs_ft.train"
output_dir="music"
output_dir+="_supervised"

model = fastText.train_supervised(corpus_file,dim=300,wordNgrams=5,loss="hs",lr=1.0,epoch=100)
model.save_model(output_dir+"_dim-300.bin")

model = fastText.train_supervised(corpus_file,dim=512,wordNgrams=5,loss="hs",lr=1.0,epoch=100)
model.save_model(output_dir+"_dim-512.bin")

model = fastText.train_supervised(corpus_file,dim=1024,wordNgrams=5,loss="hs",lr=1.0,epoch=100)
model.save_model(output_dir+"_dim-1024.bin")
