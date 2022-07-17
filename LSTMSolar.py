
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt


df = pd.read_csv (r'E:\Thesis\python_code\LSTM\Updated two years data\solar 2 years\solar one year.csv', encoding = "ISO-8859-1") 
print(df.shape)
print (df)

# creating a pandas dataframe
df = pd.DataFrame(df, columns=[
                  'Month', 'Solarirradiance'])
df['Month'] = df['Month'].astype(float)
  
 
# lets find out the data type 
# of 'Weight' column
print(df.dtypes)

#predictive file
pred_input =  pd.read_csv (r'E:\Thesis\python_code\LSTM\Updated two years data\solar 2 years\solar one year - pred.csv', encoding = "ISO-8859-1") 
print (pred_input)


X = array(df.Month).reshape(24, 1, 1)
print(X)
Y = array(df.Solarirradiance).reshape(24, 1, 1)

# Create the model

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(1, 1)))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
print(model.summary())
history=model.fit(X,Y , epochs=200, validation_split=0.3, batch_size=1)
pred_input = array([pred_input])
pred_input = pred_input.reshape((12,1, 1))
pred_output = model.predict(pred_input, verbose=0)

print(pred_output)
#print(history.history['loss'])
#print(history.history['accuracy'])
#print(history.history['val_loss'])
#print(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
