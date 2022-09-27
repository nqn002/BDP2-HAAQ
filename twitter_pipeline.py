
# Import library
import tweepy
import pandas as pd
import configparser
import pymongo
from pymongo import MongoClient
import json
from datetime import date
import csv
from time import sleep
import pandas as pd
import numpy as np
from datetime import date
import re
import spacy
import en_core_web_sm
from spacy.lang.en import English
from emoji import demojize
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Function to stream data
def twitter_collection_data(number_of_tweets, number_of_times):
    #Read credentials from config file
    config = configparser.ConfigParser()  #creating a config parser instance
    config.read('config.ini')

    api_key = config['twitter']['api_key']
    api_key_secret = config['twitter']['api_key_secret']

    access_token = config['twitter']['access_token']
    access_token_secret = config['twitter']['access_token_secret']

    #Authenticate to Twitter API
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)

    # MongoDB connection 

    #Ambi Mongo DB account
    # myclient = pymongo.MongoClient("mongodb://AmbikaNair:MongoDB@cluster0.wjpum.mongodb.net/BDP-2&retryWrites=true&w=majority?ssl=true&ssl_cert_reqs=CERT_NONE")

    # mydb = myclient["BDP-2"] #Database

    # mycol = mydb["TweetsVersion1"] 

    #Quyen MongoDB Account
    myclient = pymongo.MongoClient("mongodb+srv://HAAQ:BigDataProgramming2@cluster0.p7f2o8h.mongodb.net/?retryWrites=true&w=majority")
    # myclient = pymongo.MongoClient('mongodb://localhost:27017/')   
    # myclient = pymongo.MongoClient('mongodb+srv://adhi1041:adhi1041@cluster0.m6mzpje.mongodb.net/test')   

    mydb = myclient["BD2"] #Database

    mycol = mydb["Tweets_v1"]

    # Get today 
    today = date.today()
    file_name = ('Twitter_realtime_data_'+ str(today))
    
    #Create API
    api = tweepy.API(auth, wait_on_rate_limit=True)

    tweets_df = []
    time_df = []
    user_df = []

    # Streaming data
    for i in range(number_of_times): 
        for tweet in tweepy.Cursor(api.home_timeline, tweet_mode = "extended").items(number_of_tweets):

            tweets_df.append(tweet.full_text)
            time_df.append(tweet.created_at)
            user_df.append(tweet.user.screen_name)

            tweets = tweet.full_text
            time = tweet.created_at
            user= tweet.user.screen_name
            
            # Code sleeping to avoid over limitation
            async def main():
                await asyncio.sleep(900)

    # Dictionary format to insert to Mongo
            title_list = ["Tweets", "Created At", "User"]
            tweets_list = [tweets, time, user]
            data_dictionary = dict(zip(title_list, tweets_list))

    # Insert into database
            mycol.insert_one(data_dictionary)
    df = pd.DataFrame({'Tweets': tweets_df, 'Created At': time_df, 'User': user_df})
    df.to_csv(file_name +'.csv')
    print("script1 completed -----------")
    # return df
    
twitter_collection_data(2, 1)


################### NLP Pipeline

#Read data (data got for today from data extraction)
def read_csv_today():
    today = date.today()
    file_name = ('Twitter_realtime_data_'+ str(today))
    df=pd.read_csv(file_name +'.csv')
    return df

df=read_csv_today()

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


# Preprocessing steps:

def preprocess(input_text):
    nlp2 = en_core_web_sm.load()
    nlp = English()
    # nlp2= spacy.load('en_core_web_sm')
    tokenizer = nlp.tokenizer

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
df['cleaned_Tweets'] = df['Tweets'].apply(preprocess)
# remove the empty string values from the dataframe.
df = df[df['cleaned_Tweets'] != '']    

# Get today
today = date.today()
file_name = ('Twitter_realtime_data_cleaned_'+ str(today))
df.to_csv(file_name +'.csv')
print("script2 completed ------------ ")



##################### SENTIMENT ANALYSIS




# Read cleaned data from nlp stage 
def read_csv_cleaned_today():
    today = date.today()
    file_name = ('Twitter_realtime_data_cleaned_'+ str(today))
    df=pd.read_csv(file_name +'.csv')
    return df

df_clean = read_csv_cleaned_today()

# Modeling

def sentiment_analysis(df):
    sid=SentimentIntensityAnalyzer()
    df['sentiment_scores']= df['cleaned_Tweets'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['sentiment']= df['sentiment_scores'].apply(lambda y: 'Positive' if y>=0 else 'Negative')

    return df

df_final= sentiment_analysis(df_clean)
# df_final.sentiment.value_counts()
# print(df_clean)

# mongodb connection details
# myclient = pymongo.MongoClient('mongodb+srv://adhi1041:adhi1041@cluster0.m6mzpje.mongodb.net/test')   
myclient = pymongo.MongoClient("mongodb+srv://HAAQ:BigDataProgramming2@cluster0.p7f2o8h.mongodb.net/?retryWrites=true&w=majority")
mydb = myclient["BD2"] #Database
mycol = mydb["sentiment_class"]

df_final.reset_index(inplace=True)
df_final_dict = df_final.to_dict("records")

# insert data into mongodb collection
mycol.insert_many(df_final_dict)
print("script3 completed ------------")





