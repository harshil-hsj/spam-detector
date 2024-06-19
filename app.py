import os
import pickle
import string

import nltk
nltk.download('punkt')
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]

    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join(y)

# Exception handling for loading the pre-trained model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer not found. Make sure to train the model and save it before running this script.")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not a spam")
port = int(os.getenv('STREAMLIT_PORT', 8501))