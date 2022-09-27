# Import library
import tweepy
import pandas as pd
import pymongo
from pymongo import MongoClient
import json
from datetime import date
import csv
from time import sleep


# Function to stream data
def twitter_collection_data(number_of_tweets, number_of_times):
   
    api_key = 'P973BYkRNSPwtXss80vvsthf0'
    api_key_secret = 'JegoiiElTv3nMVRLHxI9bmG5RJzZSPWH7it9i27T62OJBHChdV'
    access_token = '1541712377430827008-RHlJcKrRZLg1AXAou5zsBVaRBSUkow'
    access_token_secret = 'VyvfFCAq7tI3H1Pzm6703Ma8wb24oIEJrPNilvfclVeIP'
    #Authenticate to Twitter API
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    # MongoDB connection
    myclient = pymongo.MongoClient("mongodb+srv://HAAQ:BigDataProgramming2@cluster0.p7f2o8h.mongodb.net/?retryWrites=true&w=majority&ssl=true")
    mydb = myclient["BD2"] #Database
    mycol = mydb["tweets_v2"]
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
    # df.to_csv(file_name +'.csv')
    print('Tweets Inserted!')
    print(df.shape)
twitter_collection_data(10,1)