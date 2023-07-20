import streamlit as st
from utils import PrepProcesor, columns 

import numpy as np
import pandas as pd
import joblib
from PIL import Image


model = joblib.load('D:/streamlit/trained_mode2.sav')
image=Image.open('D:/streamlit/titanic2.jpg')
st.title(':blue[_Titanic_] :red[Survival Prediction]',)
st.header('Did _You_ survive? :ship:')
st.image(image,use_column_width='always',clamp=False)

passengerid = st.text_input("Input Passenger ID") 
pclass = st.selectbox("Choose class", [1,2,3])
st.subheader('Enter: ')
st.text('0 for Female')
st.text('1 for Male')
sex = st.text_input("Sex:")
age = st.slider("Choose age",0,100)
sibsp = st.slider("Choose siblings",0,10)
parch = st.slider("Choose parch",0,2)
st.subheader('Select: ')
st.text('0 for Cherbourg')
st.text('1 for Queenstown')
st.text('2 for Southampton')
embarked = st.select_slider("From Where they Embark?", ['0','1','2'])

def predict(): 
    row = np.array([passengerid,pclass,sex,age,sibsp,parch,embarked]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('Passenger Survived :thumbsup:')
    else: 
        st.error('Passenger did not Survive :thumbsdown:') 

trigger = st.button('Predict',on_click=predict)

