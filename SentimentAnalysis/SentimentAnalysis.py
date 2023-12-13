#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:56:25 2023

@author: nataliamora
"""

### Sentiment Analyses

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import io
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from RateMyProfessor.rmp_api import get_schools_reviews
from DataPrep import prep_data, PreProcess


path1 = "/Course Buddy Project/"
path2 = "./RateMyProfessor/Data/"
df = pd.read_csv(path2+"RMP_university_reviews.csv", index_col = 0)
new_df = prep_data(df, n=7671)

X = np.array(new_df.Reviews)
Y = np.array(new_df.LABEL)

# Split TRAINING & TESTING data
x_train, x_valid, label_train, label_valid  = train_test_split(X, Y, test_size=0.25, random_state=69, stratify=Y) 

# Encode Labels
MyEncoder = LabelEncoder() #instantiate
y_train = MyEncoder.fit_transform(label_train)
y_valid = MyEncoder.transform(label_valid)


print("Training Data shape = ", x_train.shape)
print("Training Labels shape = ", y_train.shape)

print("Validation Data shape = ", x_valid.shape)
print("Validation Labels shape = ", y_valid.shape)

print("\n\nTraining Data Sample:\n", x_train[2])



# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
## Get Pre-trained FastText Embeddings

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

FT_embeddings = load_vectors("FastText/"+"wiki-news-300d-1M-subword.vec")
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

vocab_size = 10000 # Max number of tokens in vocabulary
seq_len = 500
embedding_dim = 300 # Dimension of FastText embeddings


### PRE-PROCESSING FUNCTION
# Tokenize Text - transforms text into sequence of integer tokens

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)


'''
# By Hand, Manual tokenizer
def tokenize_texts(texts):
    return [word_tokenize(text.lower()) for text in texts]

## Convert Tokens to Indices
def tokens_to_indices(tokenized_texts, fasttext_embeddings, vocab_size):
    indices_texts = []
    for text in tokenized_texts:
        indices_text = [fasttext_embeddings.get(word, 0) for word in text]
        indices_text = [idx if idx < vocab_size else 0 for idx in indices_text]  # Replace out-of-vocab indices
        indices_texts.append(indices_text)
    return indices_texts

def PreProcess(data, embeddings, vocabsize=vocab_size, maxlen=seq_len):
    tokenized = tokenize_texts(data)
    indices = tokens_to_indices(tokenized, FT_embeddings, vocabsize)
    padded = pad_sequences(indices, maxlen, padding='post') # PAD Sequences
    return padded
'''
   
### CREATE EMBEDDING MATRIX
# Weights for LSTM embedding layer (to implement pre-trained embeddings)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, index in tokenizer.word_index.items():
    if index < vocab_size:
        embedding_vector = FT_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = np.array(embedding_vector, dtype=np.float32)


### SET UP MODEL
LSTM_Model = tf.keras.models.Sequential([
    # Use  pre-trained FastText embeddings
    tf.keras.layers.Embedding(input_dim=vocab_size,
                              output_dim=embedding_dim,
                              weights=[embedding_matrix],
                              input_length=seq_len,
                              trainable=False),
    # Long Short Term Memory
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50, kernel_regularizer='l2', return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50)),
    tf.keras.layers.Dropout(0.2), # Randomly turns 20% of input units to 0 (prevents overfitting)
    tf.keras.layers.Dense(1, activation= 'sigmoid')
    ])

LSTM_Model.summary()

### Compile and then train model
LSTM_Model.compile(
    loss = "binary_crossentropy",
    metrics=["accuracy"],
    optimizer='adam' # robust, applicable to wide range of problems
    )


# PRE-PROCESS DATA
# Tokenize & Pad
x_train_padded = PreProcess(x_train, tokenizer)
x_valid_padded = PreProcess(x_valid, tokenizer)

### TRAIN MODEL ON DATA
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5, verbose=1, mode='max', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

### FIT THE MODEL
model_train = LSTM_Model.fit(x_train_padded, y_train,
                        epochs = 12,
                        batch_size = 100,
                        validation_data = (x_valid_padded, y_valid),
                        callbacks=[early_stopping, model_checkpoint]
                        )


# Visualize training results
# Plot Accuracy over epochs
plt.figure()
plt.plot(model_train.history['accuracy'], label="Training Data Accuracy", color='green')
plt.plot(model_train.history['val_accuracy'], label="Validation Data Accuracy", color='purple')
plt.ylabel('Accuracy')
plt.xlabel("Epochs")
plt.legend()

# Plot Loss over epochs
plt.figure()
plt.plot(model_train.history['loss'], label="Training Data Loss", color='blue')
plt.plot(model_train.history['val_loss'], label="Validation Data Loss", color='pink')
plt.ylabel('Categorical Cross Entropy Loss')
plt.xlabel("Epochs")
plt.legend()



### TEST MODEL
# Get some testing data (reviews from school not in training set)

school = "California State University San Marcos"
test_df = get_schools_reviews(school, output="dataframe")

test_df = prep_data(test_df, n=109)
x_test = np.array(test_df.Reviews)
label_test = np.array(test_df.LABEL)

# Encode Labels
y_test = MyEncoder.transform(label_test)

# Tokenize & Pad
x_test_padded = PreProcess(x_test, tokenizer)

### EVALUATE MODEL
test_loss, test_accuracy = LSTM_Model.evaluate(x_test_padded, y_test)

predictions = LSTM_Model.predict([x_test_padded])
print("\nThe prediction vector for the first data point in the test data is:\n", predictions[0])
#print("\nThe index of the maximum value in the vector gives a predicted label of:", np.argmax(predictions[0]))
#print("\nThe actual label in the test set labels is:", y_test[0])

# Get just the numeric label predictions for test data
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
predicted_labels = np.squeeze(predictions)

## Pretty Confusion Matrix
lab_dict= { 
   0:"negative", 
   1:"positive"
   }

cm = confusion_matrix(y_test, predicted_labels, labels=list(lab_dict.keys()))
print(cm)

fig, ax = plt.subplots(figsize=(13,13)) 
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
#annot=True to annotate cells, ftm='g' to disable scientific notation
# annot_kws si size  of font in heatmap
# labels, title and ticks
ax.set_xlabel('True Rating') 
ax.set_ylabel('Predicted Rating')
ax.set_title('Confusion Matrix: RNN') 
ax.xaxis.set_ticklabels(list(lab_dict.values()), fontsize = 18)
ax.yaxis.set_ticklabels(list(lab_dict.values()), rotation=90, fontsize = 18)



# Explore Incorrectly Classified Images
i = 0
for b in (y_test == predicted_labels):
    if b == False:
        print(i)
        print(y_test[i], predicted_labels[i])
    i += 1

i = 217
x_test[i]


# Commented out to avoid replacing existing models
#pickle.dump(LSTM_Model, open("SentimentAnalysesModel.pkl", 'wb'))
#pickle.dump(tokenizer, open("SentimentAnalysesTokenizer.pkl", 'wb'))

