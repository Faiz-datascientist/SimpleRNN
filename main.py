import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#load IMDBa dataset
word_index=imdb.get_word_index()
reverse_word_index={value: key for key,value in word_index.items()}

#load the model
model=load_model('simpleRNN.h5')

#Step 2: helper function 
max_length=500
def decoded(encoded_reviews):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_reviews])

#function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_reviews = [word_index.get(word,2)+3 for word in words]
    padding_review=pad_sequences([encoded_reviews],maxlen=max_length,padding='pre')
    return padding_review

#Step3 Creating prediction function
def predict_sentiment(review):
    processed_input=preprocess_text(review)
    prediction = model.predict(processed_input)
    sentiment= 'Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]



#Streamlit App
import streamlit as st

st.title("IMDB feedback review")
st.write("Enter a moview review to classify its positive or negative")


#user input
user_input=st.text_area('Moview Review')
if st.button('classify'):
    preprocessed_input=preprocess_text(user_input)


    #prediction 
    prediction=model.predict(preprocessed_input)
    sentiment='positive' if prediction[0][0]> 0.5 else 'Negative'
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]}")
else:
    st.write("Please enter a review and click classify")