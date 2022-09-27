# Import the library
import io
import os
import sys
import yaml
import pickle
import random 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


def featurize(data, output, max_len, max_words):
    
    print("Shape : "+str(data.shape)) 
    
    # Create predictors and class variables for train, test and valid sets
    X = data.clean_comments
    Y = data.sentiment

    # Tokenize
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X)

    #sequences
    sequences = tok.texts_to_sequences(X)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    
    with open(output, "wb") as fd:
        pickle.dump((sequences_matrix, Y, max_len, max_words), fd)
    pass

############################ Mainline proessing starts here  ############################ 

params = yaml.safe_load(open("params.yaml"))["featurize"]
seed = params["seed"]
random.seed(seed)

if len(sys.argv) != 4:
    len_sys = len(sys.argv)
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.stderr.write("\tExpected {} input arguments\n".format(str(4)))
    sys.stderr.write("\tReceived {} arguments\n".format(str(len_sys)))
    sys.exit(1)

# Read raw twitter data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])

# Configure output directory and file path
os.makedirs(sys.argv[3], exist_ok=True)
output_train = os.path.join(sys.argv[3], "train.pkl")
output_test = os.path.join(sys.argv[3], "test.pkl")

# Calculate maximum number of words and sequence length
if train["clean_comments"].str.len().max() >= test["clean_comments"].str.len().max():
    max_len = train["clean_comments"].str.len().max()
else:
    max_len = test["clean_comments"].str.len().max()

if train["clean_comments"].str.split("\\s+").str.len().max() >= test["clean_comments"].str.split("\\s+").str.len().max():
    max_words = train["clean_comments"].str.split("\\s+").str.len().max()
else:
    max_words = test["clean_comments"].str.split("\\s+").str.len().max()

print("Maximum length of sequence : "+str(max_len))
print("Maximum number of words : "+str(max_words))


# Limit the number of rows processed
print("Featurizing train data")
featurize(train, output_train, max_len, max_words)

print("Featurizing test data")
featurize(test, output_test, max_len, max_words)



