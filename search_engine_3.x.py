#!/usr/bin/python3
#coding: utf-8

# Purpose: skeleton for the TextIR project
#
# Comment: parts to be completed or modified are denoted with '???'
#

# Code:

##########################################################################
#                            INITIALIZATION                              #
##########################################################################

# ./search_engine_3.x.py -c cisi.all -q cisi.qry -s common_words.total_en.u8.txt -o toto
import os, codecs, sys, glob, re, getopt, random, operator
from math import *
from collections import defaultdict


prg = sys.argv[0]
def P(output=''): input(output+"\nDebug point; Press ENTER to continue")
def Info(output='',ending='\n'): #print(output, file=sys.stderr)
        sys.stderr.write(output+ending)


#######################################
# special imports



#######################################
# files


#######################################
# variables



#########################################



#########################################
# USAGE - this part reads the command line

# typical call: search_engine.py -c cisi.all -q cisi.qry -o run1

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-c", "--collection", dest="file_coll",
                  help="file containing the docs", metavar="FILE")
parser.add_argument("-q", "--query", dest="file_query",
                  help="FILE contains queries", metavar="FILE")

# ??? update the path to the stop-word list here if needed
parser.add_argument("-s", "--stop", dest="file_stop", 
                  default='./common_words.total_en.u8.txt',  
                  help="FILE contains stop words", metavar="FILE")

parser.add_argument("-o", "--out", dest="prefix",
                  help="PREFIX for output files", metavar="STR")

parser.add_argument("-v", "--verbose",
                  action="store_false", dest="verbose", default=True,
                  help="print status messages to stdout")



args = parser.parse_args()

# command line arguments are named as  args.file_coll   args.file_query ...


################################################################################
################################################################################
##                                                                            ##
##                                 FUNCTIONS                                  ##
##                                                                            ##
################################################################################
################################################################################

def Tokenizer(sequence):
	# This is a basic tokenizer which can be improved
    t_words = []
    # ??? transform a sequence as a list of words (or stems)
    # useful: line.split('...')  or better: re.split('...',line)
    # useful: line.lower()
    t_words = sequence.lower().split(' ')

    return t_words


################################################################################
################################################################################
##                                                                            ##
##                                     CODE                                   ##
##                                                                            ##
################################################################################
################################################################################


Info('Reading stop words')

# useful: line = line.rstrip('\r\n') # remove the carriage return 

t_stop_word=list()
with open(args.file_stop,'r') as f:
    for line in f:
        word = line.rstrip('\r\n')
        t_stop_word.append(word)

#list_stop_word = [line.rstrip('\r\n') for line in open(args_file_stop)]
#####################################################################

Info('Reading/indexing the collection file')


# ??? read and process the collection file to build the inverted file
# and collect any useful information (for TF-IDF/cosine or Okapi BM-25 or other models)
nb_doc = 0
h_inverted=defaultdict(lambda: defaultdict (lambda:0))   # dict of dict initialize ()
with open(args.file_coll,'r') as f:
    for line in f:
        m=re.search('^\.I ([0-9]+)',line)
        if m is not None:
            doc_id = m.group(1)
            nb_doc += 1
            
        else:
            for word in Tokenizer(line.rstrip('\r\n')):
                h_inverted[word][doc_id] +=1




#####################################################################

Info('Post-processing the inverted file')

# ??? filter out unwanted tokens in the inverted file
# compute IDF of terms (if TF-IDF is used)...
# useful: log(x)

# compute norms of documents (if cosine similarity is used)...
#useful: sum([(x*y)**2  for x in t_toto ])

for w in t_stop_word:
    if w in h_inverted: 
        del h_inverted[w]


#IDF
# number of document containing the word "word" is:

h_IDF = dict()
for w in h_inverted:
    n_w_d =len(h_inverted[word])
    h_IDF[w]= log(nb_doc/n_w_d)


h_norm = defaultdict(lambda:0)

for w in h_inverted:
    for d in h_inverted[w]:
        h_norm[d] += (h_inverted[w][d]*h_IDF[w])**2


for d in h_norm:
    h_norm[d]=sqrt(h_norm[d]) 
        



#####################################################################


Info('Reading query file')


# dictionary query -> document -> score of document for this query
h_qid2did2score = defaultdict(lambda : defaultdict(lambda : 0))
# ??? read and process the queries and keep the results in a dictionary h_qid2did2score

with open(args.file_query,'r') as f:
    for line in f:
        m=re.search('^\.I ([0-9]+)',line)
        if m is not None:
            qid = m.group(1)            
        else:
            for w in Tokenizer(line.rstrip('\r\n')):
                for d in h_inverted[w]:
                    h_qid2did2score[qid][d] += (1 * h_inverted[w][d] * h_IDF[w]) /h_norm[d]
                    
                    

    



# output the results with the expected results in a file
resultFile = open(args.prefix+'.res','w')

for qid in sorted(h_qid2did2score, key=int): # tri par numero de requete
    for (rank,(did,s)) in enumerate(sorted(h_qid2did2score[qid].items(), key=lambda t_doc_score:(-t_doc_score[1],t_doc_score[0]) ) ): # tri par score decroissant 
        resultFile.write(str(qid)+'\tQ0\t'+str(did)+'\t'+str(rank+1)+'\t'+str(s)+'\tExp\n')

resultFile.close()
