# Import the library 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Activation, Dense
from tensorflow.keras.layers import Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# Split to train and test 
X = df.clean_comments
Y = df.sentiment
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

#Paramater
max_len = df["clean_comments"].str.len().max()
max_words= df["clean_comments"].str.split("\\s+").str.len().max()
#Totken 
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
#sequences
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# RNN function 
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer) # two labels 
    model = Model(inputs=inputs,outputs=layer)
    return model

#Run empty box 
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

# Train model
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,validation_split=0.2,
callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

#Accuracy 
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n Loss: {:0.3f}\n Accuracy: {:0.3f}'.format(accr[0],accr[1]))
