# Import the library 
import io
import os
import sys
import yaml
import pickle
import random 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Activation, Dense
from tensorflow.keras.layers import Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping


# RNN function 
def RNN(max_words, max_len, activation_dense, dropout, activation_classification):
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation(activation_dense)(layer)
    layer = Dropout(dropout)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation(activation_classification)(layer) # two labels
    model = Model(inputs=inputs,outputs=layer)
    
    return model


def train_model(params, train_matrix, Y_train, train_max_len, train_max_words):
    
    # Extract all parameters
    seed = params["seed"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    activation_classification = params["activation_classification"]
    activation_dense = params["activation_dense"]
    validation_split = params["validation_split"]
    dropout = params["dropout"]
    loss = params["loss"]
    metrics = params["metrics"]
      

    # Define model 
    model = RNN(train_max_words, train_max_len, activation_dense, dropout, activation_classification)
    summary = model.summary()
    print(summary)
    
    model.compile(loss=loss,optimizer=RMSprop(),metrics=metrics)

    # Train model
    model.fit(train_matrix,
              Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split,
              callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    
    return model

############################ Mainline proessing starts here  ############################ 

params = yaml.safe_load(open("params.yaml"))["train"]


if len(sys.argv) != 2:
    len_sys = len(sys.argv)
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.stderr.write("\tExpected {} input arguments\n".format(str(2)))
    sys.stderr.write("\tReceived {} arguments\n".format(str(len_sys)))
    sys.exit(1)

# Read input
train_input = sys.argv[1]
with open(train_input, "rb") as fd:
    train_matrix, Y_train, train_max_len, train_max_words = pickle.load(fd)
 
# Configure output directory and file path
#os.makedirs(sys.argv[2], exist_ok=True)
#output_model = os.path.join(sys.argv[2], "model.pkl")

# Train the model and write to output
model = train_model(params, train_matrix, Y_train, train_max_len, train_max_words)
with open("model.pkl", "wb") as fd:
    pickle.dump(model, fd)


