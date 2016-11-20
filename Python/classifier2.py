# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:41:55 2016

@author: Akshay
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:55:39 2016

@author: Akshay
"""

# Load boilerplate text
import os as os
import numpy as np
import pandas as p
from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords
import spacy.en as sp
from nltk.tokenize import word_tokenize, sent_tokenize
import re as re
import ast
import sklearn.linear_model as lm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics,preprocessing,cross_validation
from sklearn import ensemble

os.chdir('C:/Users/aksha/OneDrive/Documents/GitHub/Evergreen-or-Ephemeral/Original Dataset')
stopwords = sp.STOPWORDS
# Word stemmer
def stemming(words_l, lang="english", encoding="utf8"):
    l = []
    wnl = WordNetLemmatizer()
    for word in words_l:
        l.append(wnl.lemmatize(word))
    return l

# URL cleaner
def url_cleaner(url, stemmer_type="WordNetLemmatizer"):
        strip_list=['http', 'https', 'www', 'com', 'net', 'org', 'm', 'html', 'htm']
        url_list=[x for x in word_tokenize(" ".join(re.findall(r'\w+', url, flags = re.UNICODE | re.LOCALE)).lower()) if x not in strip_list and not x.isdigit() and x not in stopwords]
        return " ".join(stemming(url_list))
        
# String and tokenize
def preprocess_pipeline(str, stemmer_type="WordNetLemmatizer"):
    l = []
    words = []
	
    # Tokenize
    sentences=[word_tokenize(" ".join(re.findall(r'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) 
        for t in sent_tokenize(str.replace("'", ""))]
	
    for sentence in sentences:
        # Remove stopwords
        words = [w for w in sentence if w.lower() not in stopwords]
		
        # Stem words
        words = stemming(words, stemmer_type)
		
        # Convert to string
        l.append(" ".join(words))
        
    return " ".join(l)

# Boilerplate extractor
def extract_boilerplate(str, stemmer_type="WordNetLemmatizer"):
    # Adjust 'null' and extract json
    try:
        json=ast.literal_eval(str)
    except ValueError:
        json=ast.literal_eval(str.replace('null', '"null"'))

    if 'body' in json and 'title' in json:
        return (preprocess_pipeline(json['title'], stemmer_type), preprocess_pipeline(json['body'], stemmer_type))
    elif 'body' in json:
        return (" ", preprocess_pipeline(json['body'], stemmer_type))
    elif 'title' in json:
        return (preprocess_pipeline(json['title'], stemmer_type), " ")
    else:
        return (" ", " ")

print("Loading text")
traindata_raw = list(np.array(p.read_table('train.tsv'))[:,2])
testdata_raw = list(np.array(p.read_table('test.tsv'))[:,2])
y = np.array(p.read_table('train.tsv'))[:,-1]
y=y.astype(int)

print("Load URLs")
train_url_raw = list(np.array(p.read_table('train.tsv'))[:,0])
test_url_raw = list(np.array(p.read_table('test.tsv'))[:,0])

print("Load avglinksize")
train_avglinksize = list(np.array(p.read_table('train.tsv'))[:,3])
test_avglinksize = list(np.array(p.read_table('test.tsv'))[:,3])

# Preprocess URLs
print("Preprocessing URLs")
train_url = []
test_url = []
for observation in train_url_raw:
        train_url.append(url_cleaner(observation, "WordNetLemmatizer"))
for observation in test_url_raw:
        test_url.append(url_cleaner(observation, "WordNetLemmatizer"))

# Preprocess boilerplate
print("Preprocessing boilerplate")
train_body = []
test_body = []
train_title = []
test_title = []
for observation in traindata_raw:
        a, b=extract_boilerplate(observation, "WordNetLemmatizer")
        train_title.append(a)
        train_body.append(b)
for observation in testdata_raw:
        a, b=extract_boilerplate(observation, "WordNetLemmatizer")
        test_title.append(a)
        test_body.append(b)

# Group three data sets
comboX_train=[train_avglinksize[x]+' '+train_url[x]+' '+ train_title[x]+' '+train_body[x] for x in range(len(train_body))]
comboX_test=[test_avglinksize[x]+' '+test_url[x]+' '+ test_title[x]+' '+test_body[x] for x in range(len(test_body))]
        
#train_avglinksize[x]+ ' '+             

tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
grouped=comboX_train+comboX_test
lentrain=len(comboX_train)
print("fitting pipeline")
tfv.fit(grouped)
grouped=tfv.transform(grouped)
print("transforming data")
comboX_train=grouped[:lentrain]
comboX_test=grouped[lentrain:]


# Create logit parameters
log = lm.LogisticRegression(fit_intercept=True)
#log = ensemble.AdaBoostClassifier()
print("20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(log, comboX_train, y, cv=20, scoring='roc_auc')))


print("training on full data")
log.fit(comboX_train,y)


pred_train = log.predict_proba(comboX_train)[:,1]
pred_test = log.predict_proba(comboX_test)[:,1]

print("complete")

# Write out files
train_lab = p.read_csv('train.tsv', sep="\t", na_values=['?'], index_col=1)
pred_file = p.DataFrame(pred_train, index=train_lab.index, columns=['label'])
pred_file.to_csv('train sans common words.csv')

test_lab = p.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
test_file = p.DataFrame(pred_test, index=test_lab.index, columns=['label'])
test_file.to_csv('real sans common words.csv')