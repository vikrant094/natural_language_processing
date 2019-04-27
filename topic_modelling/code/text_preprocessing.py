#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import spacy
import nltk
nlp = spacy.load('en')
stops = spacy.lang.en.stop_words.STOP_WORDS
from nltk.stem import WordNetLemmatizer


def normalize(comment, lowercase =  True ,remove_stopwords = True,lemmatization = True):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        if lemmatization:
            lemma = word.lemma_.strip()
            if lemma:
                if not remove_stopwords or (remove_stopwords and word not in stops):
                    lemmatized.append(lemma)
        else:
            lemmatized.append(word)
    return lemmatized


def nltk_cleaning(comment, lowercase = True ,remove_stopwords = True, stemming =False,lemmatization = False):
    if lowercase:
        comment = comment.lower()
    
    comment =  nltk.word_tokenize(comment)
    stopwords = nltk.corpus.stopwords.words('english')+['.',"'",',','’','?','%','"','“','”']
#     len(stopwords)
    if remove_stopwords:
        comment_cleaned = [word for word in comment if word.lower() not in stopwords]
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        comment_cleaned = [stemmer.stem(word) for word in comment_cleaned ]
    if lemmatization:
        lemm = WordNetLemmatizer()
        comment_cleaned = [lemm.lemmatize(word) for word in comment_cleaned ]
    return ' '.join(comment_cleaned)


def text_preprocessing(train_df, col,lowercase = True ,remove_stopwords = True, stemming =True,lemmatization = True):
    train_df['nltk_after_clean'] = train_df[col].apply(nltk_cleaning,  lowercase=True, remove_stopwords=True)
    return train_df

