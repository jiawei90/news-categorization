# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:04:47 2018

@author: Team X
"""
import numpy as np
import pandas as pd
import json
import os
import string
import pprint
import csv

import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure

from itertools import chain
import itertools

import logging
import gensim 
from gensim import corpora

import re
# =============================================================================
# READ DATA
# COMBINE ALL ARTICLES
# PLEASE ENSURE THAT THE DIRECTORIES ARE SET ACCURATELY
# =============================================================================
with open(r'C:\Users\chee.jw\Dropbox\NUS\Semester 2\KE 5205\CA\Submission\1_All News Articles\0-news-article.json', encoding='utf-8') as f:
    news0 = json.load(f)
with open(r'C:\Users\chee.jw\Dropbox\NUS\Semester 2\KE 5205\CA\Submission\1_All News Articles\1-news-article.json', encoding='utf-8') as f:
    news1 = json.load(f)
with open(r'C:\Users\chee.jw\Dropbox\NUS\Semester 2\KE 5205\CA\Submission\1_All News Articles\2-news-article.json', encoding='utf-8') as f:
    news2 = json.load(f)   
with open(r'C:\Users\chee.jw\Dropbox\NUS\Semester 2\KE 5205\CA\Submission\1_All News Articles\3-news-article.json', encoding='utf-8') as f:
    news3 = json.load(f)    
with open(r'C:\Users\chee.jw\Dropbox\NUS\Semester 2\KE 5205\CA\Submission\1_All News Articles\4-news-article.json', encoding='utf-8') as f:
    news4 = json.load(f)   
with open(r'C:\Users\chee.jw\Dropbox\NUS\Semester 2\KE 5205\CA\Submission\1_All News Articles\5-news-article.json', encoding='utf-8') as f:
    news5 = json.load(f)

news = list(chain(*[news0,news1,news2,news3,news4,news5]))

del news0, news1, news2, news3, news4, news5

news_backup = news.copy()
news[:] = [d for d in news if d.get('content') != False]
myNews = pd.DataFrame(news)


# =============================================================================
# CONFIGURATION
# Named Entity Recognition
# =============================================================================

java_path = 'C:\\Program Files\\Java\\jre1.8.0_181\\bin\\java.exe'
os.environ['JAVAHOME'] = java_path

# Using Stanford NER Tagger
# set the path for NER tagger: the jar file and the model
#ner_model_path = 'C:\\Users\\jiawe\\Desktop\\NUS\\Stanford NLP\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\classifiers\\english.all.3class.distsim.crf.ser.gz'
#ner_jar_path = 'C:\\Users\\jiawe\\Desktop\\NUS\\Stanford NLP\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\stanford-ner.jar'

ner_model_path = 'C:\\Users\\chee.jw\\Desktop\\NUS matters\\Stanford NLP\\stanford-ner-2018-02-27\\classifiers\\english.all.3class.distsim.crf.ser.gz'
ner_jar_path = 'C:\\Users\\chee.jw\\Desktop\\NUS matters\\Stanford NLP\\stanford-ner-2018-02-27\\stanford-ner.jar'


st_ner = StanfordNERTagger(ner_model_path, ner_jar_path)

# =============================================================================
# PART 1A: NER TAGGING
# PROCESS: TO PERFORM NER ON ALL 20,679 ARTICLES 
# RUNNING THIS IS OPTIONAL
# =============================================================================
toAdd = []
length = []
count = 0

for item in myNews["content"]:
    named_entity = st_ner.tag(word_tokenize(item))
    toAdd.append(named_entity)
    length.append(len(named_entity))
    print (count)
    count = count + 1   
    
toAdd2 = []
count = 0

for item in toAdd:
    test = pd.DataFrame(item, columns=["token","Named Entity"])
    toAdd2.append(test)

toAdd3 = []
count = 0
for item in length:
    test = list(np.repeat(count, item))
    toAdd3.append(test)
    count = count + 1

flat_list = [item for sublist in toAdd3 for item in sublist]
flat_list = pd.DataFrame(flat_list)
flat_list.columns = ["my index"]

# EXPORT NECESSARY FILES SO THAT WE NEED NOT RUN THE NER PROCESS AGAIN
df = pd.concat(toAdd2, keys=range(len(toAdd2)))
myOutput = df.copy()
myOutput["myindex"] = flat_list["my index"].values
myOutput.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Interim Data\\1_tagged tokens.csv', sep=',', encoding='utf-8')

# =============================================================================
# PART 1B: NER COMBINING INTO TOKENS
# READ INTERIM FILE CONTAINING NAMED ENTITY RECOGNITION 
# PROCESS: COMBINE TOKENS BY "_" BASED ON THEIR NAMED ENTITIES
# RUNNING THIS IS OPTIONAL
# =============================================================================

myNerOutput = pd.read_csv('C:\\Users\\chee.jw\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\4_Interim Data\\1_tagged tokens.csv', sep=',', encoding='utf-8')
len(set(myNerOutput['myindex']))
set(myNerOutput['Named Entity'])
myNerOutput.columns = ['1st index', '2nd index', 'token', 'Named Entity', 'my index' ]

myNerOutput_backup = myNerOutput.copy()
myNerOutput['token'].fillna('/', inplace = True)
myNerOutput.isnull().sum()
myNerOutput_backup.isnull().sum()

numRow = len(myNerOutput)
toAdd = []
i = -1

while i < numRow-1:
    i = i + 1
    end_index = i
    
    if myNerOutput['Named Entity'][i] == 'LOCATION':
        start = i + 1
        for j in range(start, numRow):
            print(myNerOutput['Named Entity'][j])
            if myNerOutput['Named Entity'][j] == 'LOCATION':
                pass
            else:
                end_index = j
                break
        toPush = myNerOutput['token'][i]
        for count in range(i+1,end_index):
            toPush = toPush + "_" + myNerOutput['token'][count]
        toAdd.append({'token': toPush, 'article no.': myNerOutput['my index'][i], 'original index': myNerOutput['2nd index'][i], 'named entity': myNerOutput['Named Entity'][i]})
        i = end_index
    
    elif myNerOutput['Named Entity'][i] == 'PERSON':
        start = i + 1
        for j in range(start, numRow):
            if myNerOutput['Named Entity'][j] == 'PERSON':
                pass
            else:
                end_index = j
                break
        toPush = myNerOutput['token'][i]
        for count in range(i+1,end_index):
            toPush = toPush + "_" + myNerOutput['token'][count]       
        toAdd.append({'token': toPush, 'article no.': myNerOutput['my index'][i], 'original index': myNerOutput['2nd index'][i], 'named entity': myNerOutput['Named Entity'][i]})
        i = end_index
        
    elif myNerOutput['Named Entity'][i] == 'ORGANIZATION':
        start = i + 1
        for j in range(start, numRow):
            if myNerOutput['Named Entity'][j] == 'ORGANIZATION':
                pass
            else:
                end_index = j
                break
        toPush = myNerOutput['token'][i]
        for count in range(i+1,end_index):
            toPush = toPush + "_" + myNerOutput['token'][count]       
        toAdd.append({'token': toPush, 'article no.': myNerOutput['my index'][i], 'original index': myNerOutput['2nd index'][i], 'named entity': myNerOutput['Named Entity'][i]})
        i = end_index
        
    else:
        toPush = myNerOutput['token'][i]
        toAdd.append({'token': toPush, 'article no.': myNerOutput['my index'][i], 'original index': myNerOutput['2nd index'][i], 'named entity': myNerOutput['Named Entity'][i]})
         
toAdd = pd.DataFrame(toAdd)
toAdd.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Interim Data\\2_combined tokens by NER.csv', sep=',', encoding='utf-8')

# =============================================================================
# PART 1C: NER COMBINING INTO ARTICLES 
# READ INTERIM FILE WITH EDITED (COMBINED) TOKENS  
# PROCESS: MERGE EDITED TOKENS BACK TO THEIR RESPECTIVE ARTICLE 
# RUNNING THIS IS OPTIONAL
# =============================================================================

myText = pd.read_csv('C:\\Users\\chee.jw\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\4_Interim Data\\2_combined tokens by NER.csv', sep=',', encoding='utf-8')
len(set(myText['article no.']))

uniquePerson = myText.loc[myText['named entity'] == "PERSON"]
uniquePerson = uniquePerson.drop_duplicates(subset=['token'], keep = False)

uniqueLocation = myText.loc[myText['named entity'] == "LOCATION"]
uniqueLocation = uniqueLocation.drop_duplicates(subset=['token'], keep = False)

uniqueOrg = myText.loc[myText['named entity'] == "ORGANIZATION"]
uniqueOrg = uniqueOrg.drop_duplicates(subset=['token'], keep = False)

numRow = len(myText)
toAdd = []
i = -1
toPush = ""

while i < numRow-1:
    i = i + 1        
        
    if i == 0:
        toPush = myText['token'][i]
        
    elif myText['article no.'][i] == myText['article no.'][i-1]:
        toPush = toPush + " " + myText['token'][i]
    
    else:
        toAdd.append(toPush)
        toPush = myText['token'][i]
    
    if i % 100000 == 0:
        print (str(round(i/numRow*100,2))+ "% completed")

toAdd.append(toPush)
toAdd = pd.DataFrame(toAdd)

toAdd.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Interim Data\\3_news articles after NER.csv', sep=',', encoding='utf-8')

# =============================================================================
# PART 2A: BIGRAMS 
# READ INTERIM FILE WITH EDITED ARTICLES AFTER NER
# PROCESS: PRE PROCESS THE FILE FOR BIGRAMS ANALYSIS - COMBINE ALL ARTILES INTO SINGLE STRING
# RUNNING THIS IS OPTIONAL
# =============================================================================

edited_news = pd.read_csv('C:\\Users\\chee.jw\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\4_Interim Data\\3_news articles after NER.csv', sep=',', encoding='utf-8')
edited_news.columns = ['index_2', 'content']

numRow = len(edited_news)
i = 0
toPush = edited_news['content'][0]

while i < numRow-1:
    i = i + 1        
    toPush = toPush + '\n' + '\n' + edited_news['content'][i]
    
    if i%2000 == 0:
        print (str(round(i/numRow*100,2)) + "% completed")

text_file = open('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Interim Data\\4_articles into single string.txt', "w", encoding='utf-8')
text_file.write(toPush)
text_file.close()

# =============================================================================
# PART 2B: COLLOCATIONS - BIGRAMS & TRIGRAMS
# READ INTERIM FILE CONTAINING A SINGLE STRING OF ALL THE ARTICLES
# PROCESS: BIGRAMS ANALYSIS + COMBINING TOKENS BASED ON BIGRAM RESULTS + COMBINING EDITED TOKENS BACK INTO ARTICLE
# RUNNING THIS IS OPTIONAL
# =============================================================================

with open ('C:\\Users\\chee.jw\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\4_Interim Data\\4_articles into single string.txt', "r", encoding='utf-8') as myfile:
    text_single=myfile.read()

# TOKENIZED AND CHANGE TO LOWER-CASE
tokens_bg = word_tokenize(text_single)
tokens_bg = [ t.lower() for t in tokens_bg ]

# Find bigram collocations (two-word phrases) from the data
# Get the top 20 collocations using the selected metrics
bcf = BigramCollocationFinder.from_words(tokens_bg)
top20 = bcf.nbest(BigramAssocMeasures.likelihood_ratio, 20)
top50 = bcf.nbest(BigramAssocMeasures.likelihood_ratio, 50)

# There has been little noise in the earlier results
# Let's filter off stopwords and anything less than two characters long.
stopset = set(stopwords.words('english'))
filter_stops = lambda w: len(w) < 3 or w in stopset or w.isdigit()
bcf.apply_word_filter(filter_stops)

# The team believes that the raw bigrams are the most suitable
top50_chisq = bcf.nbest(BigramAssocMeasures.chi_sq, 50)
top50_pmi = bcf.nbest(BigramAssocMeasures.pmi, 50)
top50_raw = bcf.nbest(BigramAssocMeasures.raw_freq, 50)

toAdd = []
for i in range(50):
    raw = top50_raw[i][0] + "_" + top50_raw[i][1]
    likelihood = top50[i][0] + "_" + top50[i][1]
    chisq = top50_chisq[i][0] + "_" + top50_chisq[i][1]
    pmi = top50_pmi[i][0] + "_" + top50_pmi[i][1]
    toAdd.append({'raw freq': raw, 'likelihood': likelihood, 'chisq': chisq, 'pmi': pmi })

toAdd = pd.DataFrame(toAdd)
toAdd.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Useful insight\\bigram comparison.csv', sep=',', encoding='utf-8')

# TRIGRAM
tcf = TrigramCollocationFinder.from_words(tokens_bg)
top50_tri = tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 50)
top50_tri_chisq = tcf.nbest(TrigramAssocMeasures.chi_sq, 50)
top50_tri_pmi = tcf.nbest(TrigramAssocMeasures.pmi, 50)
top50_tri_raw = tcf.nbest(TrigramAssocMeasures.raw_freq, 50)

toAdd = []
for i in range(50):
    raw = top50_tri_raw[i][0] + "_" + top50_tri_raw[i][1] + "_" + top50_tri_raw[i][2]
    likelihood = top50_tri[i][0] + "_" + top50_tri[i][1] + "_" + top50_tri[i][2]
    chisq = top50_tri_chisq[i][0] + "_" + top50_tri_chisq[i][1] + "_" + top50_tri_chisq[i][2]
    pmi = top50_tri_pmi[i][0] + "_" + top50_tri_pmi[i][1] + "_" + top50_tri_pmi[i][2]
    toAdd.append({'raw freq': raw, 'likelihood': likelihood, 'chisq': chisq, 'pmi': pmi })

toAdd = pd.DataFrame(toAdd)
toAdd.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Useful insight\\trigram comparison.csv', sep=',', encoding='utf-8')

# Combine tokens if they are bigrams
myText = pd.read_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Interim Data\\2_combined tokens by NER.csv', sep=',', encoding='utf-8')

# Function to check if tokens are bigrams
def checkInBigrams(first_word, second_word, myBigrams):
    numBigrams = len(myBigrams)
    j = 0
    result = False
    
    while j < numBigrams:
        getFirst = myBigrams['1st word'][j]
        if first_word == getFirst:
            getSecond = myBigrams['2nd word'][j]
            if second_word == getSecond:
                result = True
        j = j + 1    
    return result

myBigrams = top50_df
numRow = len(myText)
toAdd = []
i = 0
toPush = ""
            
while i < numRow-1:
    first_word = myText['token'][i]
    second_word = myText['token'][i+1]    
    
    if checkInBigrams(first_word, second_word, myBigrams):
        toPush = first_word + "_" + second_word
        print(str(i) + ": " + toPush)
        toAdd.append({'token': toPush, 'article no.': myText['article no.'][i], 'named entity': myText['named entity'][i]})
        i = i  + 2
    else:
        toPush = first_word
        toAdd.append({'token': toPush, 'article no.': myText['article no.'][i], 'named entity': myText['named entity'][i]})
        i = i + 1

toPush = myText['token'][i]
toAdd.append({'token': toPush, 'article no.': myText['article no.'][i], 'named entity': myText['named entity'][i]})

toAdd = pd.DataFrame(toAdd)

# convert back to article
myText2 = toAdd
numRow2 = len(myText2)
toAdd2 = []
i = -1
toPush2 = ""

while i < numRow2-1:
    i = i + 1        
        
    if i == 0:
        toPush2 = myText2['token'][i]
        
    elif myText2['article no.'][i] == myText2['article no.'][i-1]:
        toPush2 = toPush2 + " " + myText2['token'][i]
    
    else:
        toAdd2.append(toPush2)
        toPush2 = myText2['token'][i]
    
    if i % 100000 == 0:
        print (str(round(i/numRow2*100,2))+ "% completed")

toAdd2.append(toPush2)
toAdd2 = pd.DataFrame(toAdd2)
toAdd2.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Interim Data\\5_news articles after bigrams NER.csv', sep=',', encoding='utf-8')

# =============================================================================
# PART 3: ADDITION OF TAGS
# READ INTERIM DATA CONTAINING PROCESSED ARTICLES AFTER NER AND BIGRAMS
# PROCESS: ADD TAGS TO CONTENT OF EACH ARTICLE
# RUNNING THIS IS OPTIONAL
# =============================================================================

edited_news = pd.read_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Interim Data\\5_news articles after bigrams NER.csv', sep=',', encoding='utf-8')
edited_news.columns = ['index_2', 'content']

numRow = len(edited_news)    
i = 0

news[0]["tags"]
len(news[0]["tags"])


while i < numRow:
    numTags = len(news[i]["tags"])
    if numTags > 0:
        if numTags > 1:
            myTags = news[i]["tags"][0].replace(" ", "_")
            j = 1
            while j < numTags:
                myTags = myTags + ", " + news[i]["tags"][j].replace(" ", "_")
                j = j + 1
        else:
            myTags = news[i]["tags"][0]
        
        edited_news["content"][i] = edited_news["content"][i] + " " + myTags
        
    i = i + 1
    if i%1000 == 0:
        print (i)

edited_news.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Interim Data\\6_news articles after tags bigrams NER.csv', sep=',', encoding='utf-8')

# =============================================================================
# PART 4A: TOPIC MODELLING
# READ PROCESSED ARTICLES AFTER NER AND BIGRAMS ANALYSIS
# =============================================================================

edited_news = pd.read_csv('C:\\Users\\chee.jw\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\4_Interim Data\\6_news articles after tags bigrams NER.csv', sep=',', encoding='utf-8')
edited_news.columns = ['index_1', 'index_2', 'content']
 
# FUNCTION FOR PRE-PROCESSING
mystopwords = stopwords.words('english') + ["singapore", "said", "channel", "newsasia", "also", "would"]
mystopwords = mystopwords + ["n\'t", "going", "say", "get", "want", "know", "come", "see", "even", "make", "think", "like"]
mystopwords = mystopwords + ["still", "oct", "nov", "may", "might", "need", "\'re" ]
mystopwords = mystopwords + ["sep", "monday", "two", "one", "year", "really", ]
WNlemma = nltk.WordNetLemmatizer()

def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
    tokens=[ t for t in tokens if t not in mystopwords]
    tokens = [ t for t in tokens if len(t) >= 3 ]
    text_after_process=" ".join(tokens)
    return(text_after_process)

# Apply preprocessing to every document in the training set.
text = edited_news["content"]
toks_TM = text.apply(pre_process)

toAdd = []
for s in toks_TM:
    toAdd.append(s.split())

toks_TM = pd.Series(toAdd)

np.random.seed(123)
msk = np.random.rand(len(toks_TM)) < 0.9
train_toks = toks_TM[msk]
test_toks = toks_TM[~msk]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print("completed")

# =============================================================================
# PART 4B: TOPIC MODELLING
# TOPIC MODELLING BASED ON 7 TOPICS
# =============================================================================

dictionary = corpora.Dictionary(train_toks)
dictionary.filter_extremes(no_below=3, no_above=0.8)
dtm_train = [dictionary.doc2bow(d) for d in train_toks ]

%time lda = gensim.models.ldamodel.LdaModel(dtm_train, num_topics = 7,  alpha='auto',chunksize=30, id2word = dictionary, passes = 20,random_state=9876)

lda.show_topics(num_words=20)
dtopics_train = lda.get_document_topics(dtm_train)
from operator import itemgetter
top_train = [ max(t, key=itemgetter(1))[0] for t in dtopics_train ]

from collections import Counter
c = Counter( top_train )
count_of_topics = c.items()
type(count_of_topics)
type(c)

count_of_topics_train = pd.DataFrame.from_dict(c, orient='index').reset_index()
count_of_topics_train.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Useful insight\\count of articles by topics.csv', sep=',', encoding='utf-8')

train_index = train_toks.index.tolist()
train_results = pd.DataFrame({"article index": train_index, "topic": top_train})

# =============================================================================
# PART 4C: EXLORING HYPER PARAMETERS IN TOPIC MODELLING
# LOOPS 5 TO 10 TOPICS BASED ON UPPER LIMIT = 0.7/0.8
# RESULTS: 7 TOPICS MOST REASONABLE
# RUNNING THIS IS OPTIONAL
# =============================================================================

upperLimit = 0.8
allTopics = []

dictionary = corpora.Dictionary(train_toks)
dictionary.filter_extremes(no_below=3, no_above=upperLimit)
dtm_train = [dictionary.doc2bow(d) for d in train_toks ]

numTopics = 5
while numTopics < 11:
    print (numTopics)
    %time lda = gensim.models.ldamodel.LdaModel(dtm_train, num_topics = numTopics,  alpha='auto',chunksize=30, id2word = dictionary, passes = 20,random_state=9876)
    topic_list = lda.show_topics(num_topics = numTopics , num_words=20)
    allTopics.append(topic_list)
        
    numTopics = numTopics + 1
    
df = pd.DataFrame(allTopics)
df.to_csv('C:\\Users\\chee.jw\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\topic_list_tags_5 to 10_20181011_9876_seed.csv', sep=',', encoding='utf-8')

# =============================================================================
# PART 4D: EXLORING HYPER PARAMETERS IN TOPIC MODELLING
# LOOP UPPER LIMIT AND NO. OF TOPICS
# CONCLUSION: UPPER LIMIT HAS NO IMPACT ON THE DISTRIBUTION OF THE TOPICS
# RUNNING THIS IS OPTIONAL
# =============================================================================

# Prepare a vocabulary dictionary.
dictionary = corpora.Dictionary(train_toks)

upperLimit = 0.6
allTopics = []

while upperLimit < 1:
    dictionary.filter_extremes(no_below=3, no_above=upperLimit)
    dtm_train = [dictionary.doc2bow(d) for d in train_toks ]
    upperLimit = upperLimit + 0.1
    
    numTopics = 5
    while numTopics < 8:
        print (numTopics)
        %time lda = gensim.models.ldamodel.LdaModel(dtm_train, num_topics = numTopics,  alpha='auto',chunksize=30, id2word = dictionary, passes = 20,random_state=1234)
        topic_list = lda.show_topics(num_topics = numTopics , num_words=20)
        allTopics.append(topic_list)
        
        numTopics = numTopics + 1
    
df = pd.DataFrame(allTopics)
df.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\topic_list_dual_loop_5 to 7.csv', sep=',', encoding='utf-8')


# =============================================================================
# PART 5A: WORD CLOUD
# WORD CLOUD - LEMMA AND STEMMING
# =============================================================================
edited_news_before_tags = pd.read_csv('C:\\Users\\chee.jw\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\4_Interim Data\\5_news articles after bigrams NER.csv', sep=',', encoding='utf-8')
edited_news_before_tags.columns = ['index_2', 'content']

numRow = len(edited_news_before_tags)
i = 0
toPush = edited_news_before_tags['content'][0]

while i < numRow-1:
    i = i + 1        
    toPush = toPush + '\n' + '\n' + edited_news_before_tags['content'][i]
    
    if i%2000 == 0:
        print (str(round(i/numRow*100,2)) + "% completed")

text_file = open('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Interim Data\\7_articles into single string_word_cloud.txt', "w", encoding='utf-8')
text_file.write(toPush)
text_file.close()

with open ('C:\\Users\\chee.jw\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\4_Interim Data\\7_articles into single string_word_cloud.txt', "r", encoding='utf-8') as myfile:
    text_single=myfile.read()

tokens = word_tokenize(text_single)
unique = set(tokens)
uniqueList = list(set(tokens))
single=[w for w in unique if len(w) == 1 ]

tokens_nop = [ t for t in tokens if t not in string.punctuation ]
stop = stopwords.words('english') + ["singapore", "said", "channel", "newsasia", "also", "would"]
stop = stop + ["n\'t", "going", "say", "get", "want", "know", "come", "see", "even", "make", "think", "like"]
stop = stop + ["still", "oct", "nov", "may", "might", "need", "\'re" ]
stop = stop + ["sep", "monday", "two", "one", "year", "really", ]

tokens_lower=[ t.lower() for t in tokens_nop ]
tokens_nostop=[ t for t in tokens_lower if t not in stop ]

wnl = nltk.WordNetLemmatizer()
tokens_lem = [ wnl.lemmatize(t) for t in tokens_nostop ]

snowball = nltk.SnowballStemmer('english')
tokens_snow = [ snowball.stem(t) for t in tokens_lem ]

tokens_clean = [ t for t in tokens_snow if len(t) >= 3 ]
nltk.FreqDist(tokens_clean).most_common(50)

text_clean=" ".join(tokens_clean)

wc = WordCloud(background_color="white", width = 1600, height = 800).generate(text_clean)

# Display the generated image:
# the matplotlib way:
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
wc.to_file(r"C:\Users\jiawe\Dropbox\NUS\Semester 2\KE 5205\CA\Chart Output\wc_all_articles_lemma_stemming.png")

# =============================================================================
# PART 5B: WORD CLOUD
# WORD CLOUD - LEMMA ONLY
# =============================================================================

tokens = word_tokenize(text_single)
unique = set(tokens)
uniqueList = list(set(tokens))
single=[w for w in unique if len(w) == 1 ]

tokens_nop = [ t for t in tokens if t not in string.punctuation ]
stop = stopwords.words('english') + ["singapore", "said", "channel", "newsasia", "also", "would"]
stop = stop + ["n\'t", "going", "say", "get", "want", "know", "come", "see", "even", "make", "think", "like"]
stop = stop + ["still", "oct", "nov", "may", "might", "need", "\'re" ]
stop = stop + ["sep", "monday", "two", "one", "year", "really", ]


tokens_lower=[ t.lower() for t in tokens_nop ]
tokens_nostop=[ t for t in tokens_lower if t not in stop ]

wnl = nltk.WordNetLemmatizer()
tokens_lem = [ wnl.lemmatize(t) for t in tokens_nostop ]

tokens_clean = [ t for t in tokens_lem if len(t) >= 3 ]
nltk.FreqDist(tokens_clean).most_common(50)

text_clean=" ".join(tokens_clean)

wc = WordCloud(background_color="white", width = 1600, height = 800).generate(text_clean)

# Display the generated image:
# the matplotlib way:
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
wc.to_file(r"C:\Users\jiawe\Dropbox\NUS\Semester 2\KE 5205\CA\Chart Output\wc_all_articles_lemma_only.png")


# =============================================================================
# PART 6: PREDICTION
# TESTING MODEL ON UNSEEN DOCUMENTS
# =============================================================================

dtm_test = [dictionary.doc2bow(d) for d in test_toks ]
dtopics_test = lda[dtm_test]

from operator import itemgetter
top_test = [ max(t, key=itemgetter(1))[0] for t in dtopics_test ]

from collections import Counter
c_test = Counter( top_test )
count_of_topics_test = c_test.items()
type(count_of_topics)
type(c)

count_of_topics_test = pd.DataFrame.from_dict(c_test, orient='index').reset_index()
count_of_topics = count_of_topics_train.merge(count_of_topics_test, on='index', how='left')
count_of_topics.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Useful insight\\count of articles by topics_train_test.csv', sep=',', encoding='utf-8')

# =============================================================================
# PART 7: SUMMARIZE
# =============================================================================

test_index = test_toks.index.tolist()
test_results = pd.DataFrame({"article index": test_index, "topic": top_test})

topic_interest = 3
test_range = len(test_results)

def getSummary(content, tags):
    import nltk
    
    #Tokenize to sentences
    sentence_list = nltk.sent_tokenize(content)  
    
    #Generate a list of stopwords from NLTK
    stopwords = nltk.corpus.stopwords.words('english')
    
    #Tokenize to words
    wordTokenize = nltk.word_tokenize(content)
    wordTokenizeLowercase = [word.lower() for word in wordTokenize if word.isalpha()]

    #Generating the frequencies for each term
    word_frequencies = {}  
    for word in wordTokenizeLowercase:  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
                for tag in tags:
                    if word in tag:
                        #More weights are added for similar tags
                        word_frequencies[word] += 1
            else:
                for tag in tags:
                    if word in tag:        
                        #More weights are added for similar tags
                        word_frequencies[word] += 2
                    else:
                        word_frequencies[word] += 1
    
    #Get the Max number of frequency
    maximum_frequncy = max(word_frequencies.values())
    
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
       
    #Generating Sentence Scorce
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 50:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    
    import heapq
    
    #Generate the top 7 sentences
    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summaryText = ' '.join(summary_sentences)  
    
    return summaryText

i = 0
toAdd = []
while i < test_range:
    if test_results["topic"][i] == topic_interest:
        articleIndex = test_results["article index"][i]
        summarized_news = getSummary(news[articleIndex]["content"], news[articleIndex]["tags"])
        numFull = len(nltk.word_tokenize(news[articleIndex]["content"]))
        numSummarized = len(nltk.word_tokenize(summarized_news))
        toAdd.append({'article index': articleIndex, 'full news': news[articleIndex]["content"], 'summarized news': summarized_news, 'count of words: full article': numFull, 'count of words: summarized article': numSummarized })
    
    i = i + 1
    
summarized = pd.DataFrame(toAdd)
summarized.to_csv('C:\\Users\\jiawe\\Dropbox\\NUS\\Semester 2\\KE 5205\\CA\\Submission\\Useful insight\\results_summarized articles.csv', sep=',', encoding='utf-8')


# =============================================================================
# END
# =============================================================================





