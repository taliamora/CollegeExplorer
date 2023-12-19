

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Clean up reviews
def prep_data(df, labels=True, n= None, save=False, save_name='PreProcessed_Data.csv'):
    #df = df[df.Reviews.isna() == False]
    df.Reviews = df.Reviews.astype(str)
    df = df[df['Reviews'].map(len) > 10] # Remove very short reviews (1-2 words)
    
    if labels == True:
        df["LABEL"] = df["Ratings"].map(group_labels)
        df = Balance(df, n)
        df = df.sample(frac=1).reset_index(drop = True)    
    if save == True:
        df.to_csv(save_name)
    return df
        

def group_labels(score, threshold = 3):
    if score <= threshold:
        return "negative"
    if score > threshold:
        return "positive"
    
def Balance(df, n):
    labels = np.unique(df.LABEL)
    new_df = pd.DataFrame(data=None, columns=list(df.columns))
    for lab in labels:
        num_df = df[df.LABEL == lab]
        keep = num_df.sample(n, random_state=69)
        new_df = pd.concat([new_df,keep]) 
    return new_df


# Prep Data for Sentiment Analysis Model
def PreProcess(data, tokenizer, maxlen=500):
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(data)
    padded_sequences = pad_sequences(sequences, maxlen, padding='post') # PAD Sequences
    return padded_sequences
