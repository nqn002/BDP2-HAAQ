#File to use
# Install library
# !pip install pymongo
# !pip install emoji
# !pip install spacy

# Import library
import os
import pandas as pd
import numpy as np
from datetime import date
import re
import spacy
from spacy.lang.en import English
from nltk.stem.snowball import SnowballStemmer
import en_core_web_sm
from emoji import demojize
import pymongo
from pymongo import MongoClient
import dns
import ssl
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Read data (data got for today from data extraction)
def fetch_data_from_Mongo():
    client = pymongo.MongoClient("mongodb+srv://HAAQ:BigDataProgramming2@cluster0.p7f2o8h.mongodb.net/?retryWrites=true&w=majority")
    records = client.get_database('BD2').Tweets_v1
    df = pd.DataFrame.from_dict(records.find())
    return df

df=fetch_data_from_Mongo()

# Save it to csv 
today = date.today()
file_name = ('Twitter_realtime_rawdata_'+ str(today))
df.to_csv(file_name +'.csv')
      
# Data Cleaning 
def data_cleaning(df):    
    # converting the created_time column to datetime datatype.
    df['Created At'] = pd.to_datetime(df['Created At'],utc=True)
    # Remove Unnamed column
    del df['Unnamed: 0']
    # Remove duplicated
    df['dup'] = df.duplicated(subset=None, keep='first')
    # removing the duplicate columns.
    df = df[df['dup'] == False]
    # Delete duplicated column
    del df['dup']
    # Check data null or not
    df['Tweets'].isnull().values.any()
    # Remove na
    df.dropna(inplace=True)
    return df

df=data_cleaning(df)


# Data cleaning & replacing
def replace_all(text):
    
    dict_country={
' ENGGER ':' england versus germany ',
' FRAGER ':' france versus germany ',
' GERENG ':' germany versus england ',
' GERFRA ':' germany versus france ',
' GERHUN ':' germany versus hungary ',
' GERENG ':' germany versus england ',
' GERFRA ':' germany versus france ',
' ENGGER ':' germany versus england ',
' FRAGER ':' germany versus france ',
' HUNGER ':' germany versus hungary ',
' ENGVSGER ':' england versus germany ',
' FRAVSGER ':' france versus germany ',
' GERVSENG ':' germany versus england ',
' GERVSFRA ':' germany versus france ',
' GERVSHUN ':' germany versus hungary ',
' GERVSENG ':' germany versus england ',
' GERVSFRA ':' germany versus france  ',
' ENGVSGER ':' germany versus england  ',
' FRAVSGER ':' germany versus france ',
' HUNVSGER ':' germany versus hungary ',
' GERNED ':' germany versus Netherlands ',
' GERBLR ':' germany versus belarus ',
' GEREST ':' germany versus estonia ',
' GERNIR ':' germany versus northern ireland ',
' GERPOR ':' germany vs portugal ',
' NEDGER ':' germany versus Netherlands ',
' BLRGER ':' germany versus belarus ',
' ESTGER ':' germany versus estonia ',
' NIRGER ':' germany versus northern ireland ',
' PORGER ':' germany vs portugal ',
' GERVSNED ':' germany versus Netherlands ',
' GERVSBLR ':' germany versus belarus ',
' GERVSEST ':' germany versus estonia ',
' GERVSNIR ':' germany versus northern ireland ',
' GERVSPOR ':' germany vs portugal ',
' NEDVSGER ':' germany versus Netherlands ',
' BLRVSGER ':' germany versus belarus ',
' ESTVSGER ':' germany versus estonia ',
' NIRVSGER ':' germany versus northern ireland ',
' PORVSGER ':' germany vs portugal ',
' GERMANYKKKKKKKKKKKKK ':' germany ',
' GERMANYYY ':' germany ',
' GERMANYYYYY ':' germany ',
' GERMAY ':' germany ',
' Germ ':' germany ',
' Germa ':' germany ',
' German ':' germany ',
' Germani ':' germany ',
' Germania ':' germany ',
' Germanieee ':' germany ',
' Germanies ':' germany ',
' Germans ':' germany ',
' Germany ':' germany ',
' Germany- ':' germany ',
' Germany-1 ':' germany ',
' Germany-2 ':' germany ',
' Germany/ ':' germany ',
' Germany1 ':' germany ',
' Germany4 ':' germany ',
' GermanyVsHungary ':' germany ',
' Germanycomeback ':' germany ',
' Germanygirls ':' germany ',
' Germanynis ':' germany ',
' Germanys ':' germany ',
' Germanyyyy ':' germany ',
' Germanyyyyy ':' germany ',
' Germanyyyyyyyyy ':' germany ',
' Germanz ':' germany ',
' Germeny ':' germany ',
' Geramany ':' germany ',
" Germany's ":' germany ',
' germany ':' germany ',
' Germanys ':' germany ',
' TeamGermany ':' germany ',
' GERMANE ':' germany ',
' GER\s ':' germany ',
' england ':' england ',
' ENG\s ':' england ',
' TeamEngland ':' england ',
' TeamHungary ':' hungary ',
' hungary ':' hungary ',
' HUN\s ':' hungary ',
' Hungria ':' hungary ',
' TeamItaly ':' italy ',
' italy ':' italy ',
' ITA\s ':' italy ',
' Ital ':' italy ',
' itlay ':' italy ',
' TeamFrance ':' france ',
' france ':' france ',
' FRA\s ':' france ',
' TeamNetherlands ':' netherlands ',
' netherlands ':' netherlands ',
' NED\s ':' netherlands  ',
' belarus ':' belarus ',
' BLR\s ':' belarus ',
' estonia ':' estonia ',
' EST\s ':' estonia ',
' NIR\s ':' northern Ireland ',
' portugal ':' portugal ',
' POR\s ':' portugal ',
' Nether ':' netherlands  ',
' Netherland ':' netherlands  ',
' Portugalllll ':' portugal ',
' Portugals ':' portugal ',
' TeamPortugal ':' portugal ',
' GoPortugal ':' portugal ',
' Portugal ':' portugal ',
' EURO2020\s ':' eurocup ',
' EURO2020.\s ':' eurocup ',

}
    
    for i, j in dict_country.items():
        text = re.sub(i, j,text,flags=re.IGNORECASE)
    return text

df['clean_comments']=df['Tweets'].apply(replace_all).str.lower()

# Preprocessing steps:

def preprocess(input_text):
    nlp2 = en_core_web_sm.load()
    nlp = English()
    #nlp2= spacy.load('en_core_web_lg')
    tokenizer = nlp.tokenizer
    s_stemmer=SnowballStemmer(language='english')

    # replace emojis with its respective emotion
    demojized_text= demojize(input_text)
    
    # remove the @mentions fromt the text
    pattern="@\w+"
    text_mentions_removed=re.sub(pattern,'',demojized_text)
    
    # remove the web links in the text
    http_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text_http_removed = http_pattern.sub('', text_mentions_removed)
    
    # regular expression keeping only letters
    letters_only_text = re.sub("[^a-zA-Z]", " ", text_http_removed)
           
    # convert to lower case
    text_to_lower_case= letters_only_text.lower()
    
        # word tokenization
    token_list = []
    tokens = tokenizer(text_to_lower_case)
    for token in tokens:
        token_list.append(token.text)
    
    #stop words removal
    stop_words_removed_text=[]
    for tok in token_list:
        if nlp.vocab[tok].is_stop == False and len(tok.strip())!=0 and len(tok)!=1:
            stop_words_removed_text.append(tok)
#     print(stop_words_removed_text)
#      stemming
#     stemmed_text=[]
#     for word in stop_words_removed_text:
#         stemmed_text.append(s_stemmer.stem(word))
#      print(stemmed_text)
    
    #Lemmatization
    lemmatized_text=[]
    string=''
    for w in stop_words_removed_text:
        if string=='':
            string = w
        else:
            string= string + ' ' + w 
#     print(string)
    nlp_string = nlp2(string)
    for word in nlp_string:
        lemmatized_text.append(word.lemma_)
#     print(lemmatized_text)
    # converting list back to string
    return " ".join(lemmatized_text)


# the preprocessed objective is appended to the project_df dataframe.
df['clean_comments'] = df['clean_comments'].apply(preprocess)
# remove the empty string values from the dataframe.
df_clean = df[df['clean_comments'] != '']    

# Modelling 
def sentiment_analysis(df):
    sid=SentimentIntensityAnalyzer()
    df['sentiment_scores']= df['clean_comments'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['sentiment']= df['sentiment_scores'].apply(lambda y: 'Positive' if y>=0 else 'Negative')

    return df

df_final= sentiment_analysis(df_clean)
df_final.sentiment.value_counts()