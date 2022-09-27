import tweepy
import pandas as pd
import configparser
import pymongo
from pymongo import MongoClient
import json


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

#Creating an api instance
# api = tweepy.API(auth)
api = tweepy.API(auth, wait_on_rate_limit=True)

# cursor = tweepy.Cursor(api.search_tweets, q = "London", tweet_mode = "extended").items(1)
# for i in cursor:
#     print(dir(i))  

#To stream tweets based on keyword
# cursor = tweepy.Cursor(api.search_tweets, q = "London", tweet_mode = "extended").items(1)


# Getting multiple tweets with the datetime details
number_of_tweets = 200
tweets = []
time = []
user = []

# cursor = tweepy.Cursor(api.home_timeline)
for tweet in tweepy.Cursor(api.home_timeline, tweet_mode = "extended").items(number_of_tweets):
    tweets.append(tweet.full_text)
    time.append(tweet.created_at)
    user.append(tweet.user.screen_name)

# print(tweets)

#Creating a dataframe to store the tweets and time information
df = pd.DataFrame({'Tweets': tweets, 'Created At': time, 'User': user})
print(df)

# #Connect to MongoDB

# client = MongoClient(mongod_connect)
# db = client.