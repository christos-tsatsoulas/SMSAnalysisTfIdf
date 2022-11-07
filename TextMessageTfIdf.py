"""
NLP project: analysing SMS data for topic modelling using tf-idf scores

"""
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from numpy import argsort, asarray, array

#Importing the dataset and creating a Panda dataframe
raw_data= pd.read_csv("clean_nus_sms.csv")
raw_messages= raw_data["Message"].values.astype("str").tolist()

#This function translates one module's POS output into another module's POS input. Credits to Selva Prabhakaran from "Machine Learning +"
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

#Preprocessing the messages
def preprocess(raw_data, stopword_lst):
    no_punc= [re.sub(r"[^0-9a-zA-Z ]+","",raw_data[i]) for i in range(len(raw_data))]
    tokenized= [word_tokenize(i) for i in no_punc]
    no_stopwords= []
    for sublst in tokenized:
        no_stopwords.append([ele for ele in sublst if ele.lower() not in stopword_lst])
    lemmatized= []
    lemmatizer= WordNetLemmatizer()
    for sublst in no_stopwords:
        lemmatized.append([lemmatizer.lemmatize(word,pos=get_wordnet_pos(word)) for word in sublst])
    return lemmatized


#Storing preprocessed data, creating and modifying the stopwords list
stopword_lst= stopwords.words("english")
extrawords= ["haha","lol","ok","u","im","okay","hahaha","ur","2","one","oh","hey","like","le","dont","yeah","wan","still","later","sorry","hi","also","na","ya","eh","thanks","e","lor","r","n","ah","4","wat","k","leh","really","dun","lah","la"]
stopword_lst.extend(extrawords)
processed_data= preprocess(raw_messages,stopword_lst)

#Finding the 10 most frequent terms across the corpus
wordcounts= Counter([ele.lower() for sublst in processed_data for ele in sublst])
most_common= wordcounts.most_common(10)

#Calculating tf-idf scores
untokenized= [" ".join(lst).lower() for lst in processed_data]
vectorizer= TfidfVectorizer()
tfidf=vectorizer.fit_transform(untokenized)

#Finding 10 words with the highest tf-idf scores
tfidf_feature_names = array(vectorizer.get_feature_names())
importance = argsort(asarray(tfidf.sum(axis=0)).ravel())[::-1]
top_10_words= tfidf_feature_names[importance[:10]]

"""
Findings:
    
The 10 most frequent words across the corpus were:['go','get','come','time','think','see','want','know','need','take'].

The 10 words with the highest tf-idf scores were: ['go', 'get', 'come', 'time', 'see', 'call', 'home', 'think','know', 'want'].

Words such as "go", "time" and "see" suggest that many text messages are written to organise events such as meetups.

Words such as "call" and "home" suggest that users like to know when to expect their loved ones to either arrive or return home.

Words such as "know" and "want" suggest that users often request things through text messages.
"""