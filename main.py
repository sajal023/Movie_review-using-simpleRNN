import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


#load the IMDB dataset word index
word_idx=imdb.get_word_index()
rev_word_idx={value: key for key,value in word_idx.items()}

model=load_model('simple_rnn_imdb.h5')


#Helper funciton

#funct to decode reviews
def decode_review(encoded_review):
    return ' '.join([rev_word_idx.get(i-3,'?') for i in encoded_review])

#funct to preprocess the user i/p
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_idx.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

import streamlit as st
##DEsign streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter Movie Review to predict it as Positive or Negative")

#User Input
user_input=st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    #Displaying res
    st.write(f'Setiment Score: {sentiment}')
    st.write(f'Predicition score: {prediction[0][0]}')
else:
    st.write('Please enter a review')


