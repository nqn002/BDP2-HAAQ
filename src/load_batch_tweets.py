# Import required packages
import os
import re
import sys
import json
import logging
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
n=0


# Function to import batch data from csv to Mongo DB collection
def load_batch_tweets_to_mongo(config, data):
    
    #Mongo DB storage details
    url = config['data_storage']['url']
    db = config['data_storage']['db']
    collection = config['batch_data_load']['collection']

    # MongoDB connection 
    myclient = MongoClient(url)
    mydb = myclient[db]
    mycol = mydb[collection]

    tweets_df = []
    time_df = []
    user_df = []
    try:
    	# Reset data index and drop collection if exists
        data.reset_index(inplace=True)
        mycol.drop()
        # insert data into mongodb collection
        mycol.insert_many(data.to_dict("records"))
        count = mycol.count_documents({})
        print("Inserted "+ str(count)+ " records into "+str(config['batch_data_load']['collection']))
        logging.info("Inserted "+ str(count)+ " records into "+str(config['batch_data_load']['collection']))
    except Exception as ex:
        logging.error("Error while inserting records in MongoDB! \n" + str(ex))
	           
    return None


#Read credentials from config file
def read_config_file():

    config = configparser.ConfigParser()  #creating a config parser instance
    config.read('config.ini')
    
    return config
    
    
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
    
    global n
    n += 1
    print("Currently processing: ",n)
    
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


# Modelling 
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
        
        print("NLP preprocessing starts for VADER")
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
    
    
    return df_final

#Mainline method
def main():
    
    start = datetime.now()
    
    # Set change directory path
    cwd = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
    
    # Configuring logging to log the current run information
    logs_dir = os.path.join(parent_dir,'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_file_name = 'logs_batch_data_load.log'
    logs_file = os.path.join(logs_dir, log_file_name)
        
    # Configuring the File mane to logging Level
    logging.basicConfig(filename=logs_file,level=logging.INFO)
    logging.info("Batch data loading starts!")
    
    #Read config file for data extraction configuration
    config = read_config_file()
    
    # Read batch data
    batch_data_path = os.path.join(parent_dir,'data/batch_tweets.csv')
    data = pd.read_csv(batch_data_path)
    # Limit the number of rows processed
    data = data.iloc[11:30000,:]
    # Process and label sentiments in batch data
    data_final = process_tweets(data)
    
    #Load batch tweets to DB 
    msg = load_batch_tweets_to_mongo(config, data_final)
    logging.info("Batch data loaded successfully!")
    
    end =  datetime.now()
    logging.info("Time taken :" + str(end-start))
    
    return None



if __name__ == "__main__":
	
    main()
	
