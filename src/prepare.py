# Import library
import io
import os
import re
import sys
import yaml
import random
import dvc.api
import pandas as pd
   
# Split data into train and test sets
def train_test_split(df, size):
    
    shuffle_df = df.sample(frac=1)
    train_size = int(size * len(df))
    df_train = shuffle_df[:train_size]
    df_test = shuffle_df[train_size:]
    
    return df_train, df_test    

# 
def process_tweets(data_in, train_size):
	
    try:        
        print("Splitting into train, test and valid sets")
        df_train, df_test_valid = train_test_split(data_in, train_size)
        
        print("Input : "+ str(len(data_in)))
        print("Train : "+ str(len(df_train)))
        print("Test : "+ str(len(df_test_valid)))
        
    except Exception as ex:
        sys.stderr.write(f"Exception occured in process_tweets: {ex}\n")
        return None
    
    return df_train, df_test_valid

############################ Mainline proessing starts here  ############################ 

params = yaml.safe_load(open("params.yaml"))["prepare"]
num_rows = params['num_rows']

# Test data set split ratio
train_size = params["train_size"]
random.seed(params["seed"])


if len(sys.argv) != 2:
    len_sys = len(sys.argv)
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.stderr.write("\tExpected {} input arguments\n".format(str(2)))
    sys.stderr.write("\tReceived {} arguments\n".format(str(len_sys)))
    sys.exit(1)


# Read raw twitter data
#input = pd.read_csv(sys.argv[1])
with dvc.api.open("data/twitter_data.csv",repo="https://github.com/nqn002/BD2_HAAQ") as f:
	input = pd.read_csv(f)

# Configure output directory and file path
os.makedirs(sys.argv[1], exist_ok=True)
output_train = os.path.join(sys.argv[1], "train.csv")
output_test = os.path.join(sys.argv[1], "test.csv")


# Limit the number of rows processed
input = input.iloc[:num_rows,:]
train, test = process_tweets(input, train_size)
train.to_csv(output_train)
test.to_csv(output_test)


