# Import required packages
import os
import re
import sys
import logging
import tweepy
import pandas as pd
import configparser
from pymongo import MongoClient
from datetime import date, datetime
from time import sleep
import en_core_web_sm
from emoji import demojize
from spacy.lang.en import English
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Function to collect streaming data
def twitter_collection_data(config):
    
    #Twitter api key and key secret
    api_key = config['twitter']['api_key']
    api_key_secret = config['twitter']['api_key_secret']
    #Twitter access token and token secret
    access_token = config['twitter']['access_token']
    access_token_secret = config['twitter']['access_token_secret']

    #Number of tweets limit and number of runs limit
    number_of_tweets = config['data_extraction']['number_of_tweets']
    number_of_runs = config['data_extraction']['number_of_runs']
    
    #Mongo DB storage details
    url = config['data_storage']['url']
    db = config['data_storage']['db']
    collection = config['data_storage']['collection']
    
    #Authenticate to Twitter API
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)

    # MongoDB connection 
    myclient = MongoClient(url)
    mydb = myclient[db]
    mycol = mydb[collection]
	
    # Count of documents in collection before insert
    count_before = mycol.count_documents({})
    logging.info("Count of documents in collection before insert : " + str(count_before))
    
    #Create API
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    data_full = []
    
    try:
        # Streaming data
        for i in range(int(number_of_runs)):
            for tweet in tweepy.Cursor(api.home_timeline, tweet_mode = "extended").items(int(number_of_tweets)):
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
                
                # Append each record to a list to process later
                data_full.append(data_dictionary)
                
        data_df = pd.DataFrame(data_full) 
        print("Streaming data fetched sucessfully !")
        print("Shape of data : "+str(data_df.shape))
        
    except Exception as ex:
        print("Error while fetching records from twitter! " + str(ex))
        logging.error("Error while fetching records from twitter! " + str(ex))
        return ex
           
    return data_df

   
#Read data (data got until today from Mongo DB)
def fetch_data_from_Mongo(config):
	
	#Mongo DB storage details
    url = config['data_storage']['url']
    db = config['data_storage']['db']
    collection = config['data_storage']['collection']
	
	#Connect to DB 
    client = MongoClient(url)
    records = client.get_database(db)[collection]
    
    #Fetch all records
    df = pd.DataFrame.from_dict(records.find())
    
    return df


#Read data (data got until today from Mongo DB)
def save_data_to_Mongo(config, data):
	
	#Mongo DB storage details
    url = config['data_storage']['url']
    db = config['data_storage']['db']
    collection = config['data_storage']['collection']
	
	#Connect to DB 
    client = MongoClient(url)
    mycol = client.get_database(db)[collection] 
    
    print("Current rows in DB : "+str(mycol.count_documents({})))
    # Reset data index and drop collection if exists
    data.reset_index(inplace=True)
    # Insert data into mongodb collection
    result = mycol.insert_many(data.to_dict("records"))
    count = mycol.count_documents({})
    
    print("Rows after insertion : "+str(mycol.count_documents({})))
    logging.info("Inserted records : " + str(result.inserted_ids))
    
    return None


#Read credentials from config file
def read_config_file():

    config = configparser.ConfigParser()  #creating a config parser instance
    config.read('config.ini')
    
    return config
   
def extract_main():
    
    start = datetime.now()
	
	# Set change directory path
    cwd = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
    
    # Configuring logging to log the current run information
    logs_dir = os.path.join(parent_dir,'logs')
    #logs_dir = '/home/hemap/big-data-programming-2-april-2021-haaq-alwaysexecutable/logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_file_name = 'logs_data_extraction_'+str(date.today())+'.log'
    logs_file = os.path.join(logs_dir, log_file_name)
    
    # Configuring the File mane to logging Level
    logging.basicConfig(filename=logs_file,level=logging.INFO)
    
    #Read config file for data extraction configuration
    config = read_config_file()
    
    #Collect streaming tweets 
    logging.info("Data extraction starts : "+ str(start))
    data = twitter_collection_data(config)
    logging.info("Data extracted successfully!")
	
	# Pre-Process and Label streamed data
    print("Pre-Process and Label streamed data")
    df_final = process_tweets(data)
	
	# Insert into database
    print("Inserting data into Mongo DB with labels")
    save_data_to_Mongo(config, df_final)
    
    #Fetch updated data records until today from Monngo DB collection
    df=fetch_data_from_Mongo(config)
    
    #Save fetched data to csv format in data folder
    data_dir = os.path.join(parent_dir,'data')
    #data_dir = '/home/hemap/big-data-programming-2-april-2021-haaq-alwaysexecutable/data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_file = os.path.join(data_dir, 'twitter_data.csv')
    df.to_csv(data_file,index=False)
    logging.info("Data stored successfully!")
    
    end = datetime.now()
    logging.info("Time taken : "+ str(end-start))
    
    return None
    

 # Data Cleaning 
def data_cleaning(df):    
    # converting the created_time column to datetime datatype.
    df['Created At'] = pd.to_datetime(df['Created At'],utc=True)
    print("Converting the created_time column to datetime datatype")
    # Remove Unnamed column
    #del df['Unnamed: 0']
    # Remove duplicated
    df['dup'] = df.duplicated(subset=None, keep='first')
    print("Remove duplicated")
    # removing the duplicate columns.
    df = df[df['dup'] == False]
    # Delete duplicated column
    del df['dup']
    print("Delete duplicated column")
    # Check data null or not
    df['Tweets'].isnull().values.any()
    # Remove na
    df.dropna(inplace=True)
    print("Remove na")
    return df


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


# Preprocessing steps:
def preprocess(input_text):
    
    nlp2 = en_core_web_sm.load()
    nlp = English()
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
    
    #Lemmatization
    lemmatized_text=[]
    string=''
    for w in stop_words_removed_text:
        if string=='':
            string = w
        else:
            string= string + ' ' + w 
    nlp_string = nlp2(string)
    
    for word in nlp_string:
        lemmatized_text.append(word.lemma_)
    
    # converting list back to string and return
    return " ".join(lemmatized_text)


# Labelling sentiments using VADER 
def sentiment_analysis(df):
    sid=SentimentIntensityAnalyzer()
    df['sentiment_scores']= df['clean_comments'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['sentiment']= df['sentiment_scores'].apply(lambda y: 1 if y>=0 else 0)

    return df


def process_tweets(data_in):
	
    try:
        print("Converting to lower case")
        # Clean raw data
        df = data_cleaning(data_in)
        df['clean_comments']=df['Tweets'].apply(replace_all).str.lower()
        
        print("Preprocessing starts for VADER")
        # The preprocessed objective is appended to the project_df dataframe.
        df['clean_comments'] = df['clean_comments'].apply(preprocess)
        
        print("Removing the empty string values from the dataframe")
        # Remove the empty string values from the dataframe.
        df_clean = df[df['clean_comments'] != '']   
        
        print("Adding sentiment labels Negative/Positive")
        df_final = sentiment_analysis(df_clean)
        
    except Exception as ex:
        sys.stderr.write(f"Exception occured in process_tweets: {ex}\n")
        return None
    
    #return df_train, df_test, df_valid
    return df_final
    
    return None

extract_main()
