import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import string

port=PorterStemmer()


def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(port.stem(i))

    return " ".join(y)


with open('vectorizer.pkl', 'rb') as f:
    tfid = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

#tfid=pickle.load(open('vectorizer.pkl','rb'))
#model=pickle.load(open('model.pkl','rb'))

st.title('SMS SPAM DETECTION')

sms=st.text_input('Enter the msg: ')

if st.button('Predict'):

    # preprocess
    transform_sms=transform(sms)
    # vectorize
    vector_input=tfid.transform([transform_sms])
    # predict
    result=model.predict(vector_input)[0]
    # display
    if result==1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')