import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import random
import time

st.header('Startup Success Prediction Using Machine Learning')

data = '''The objective of this project is to predict the success of startups based on financial and operational features such as R&D Spend, Administration, Marketing Spend, and Location. By applying multiple machine learning models and comparing their performance, the project aims to identify key factors influencing profitability and provide insights into investment strategies for startup growth.'''

st.subheader(data)

st.image('https://www.atticusadvisors.co.in/wp-content/uploads/2024/10/1691517271641-1024x529.jpg')

# Load trained model
with open('startup_success_pred.pkl','rb') as f:
    success_prediction = pickle.load(f)

# ✅ Load dataset locally (make sure 50_Startups.csv is in your repo)
df = pd.read_csv("50_Startups.csv")

st.sidebar.header("Select Features to check your startup's success")
st.sidebar.image('https://img.jagranjosh.com/images/2022/August/1082022/what-is-a-start-up-types-funding-compressed.webp')

all_values = []

# LabelEncoder mapping (must match training)
state_mapping = {"California": 0, "Florida": 1, "New York": 2}

for i in df.iloc[:, :-1]:   # all features except target
    if df[i].dtype == 'object':   # categorical (like State)
        var = st.sidebar.selectbox(f"Select {i}", df[i].unique())
        var = state_mapping[var]   # encode state to int
    else:   # numeric
        min_value, max_value = df[i].agg(['min', 'max'])
        var = st.sidebar.slider(
            f'Select {i} value',
            float(min_value),
            float(max_value),
            float(df[i].mean())   # default = mean
        )
    all_values.append(var)

random.seed(132)
# Convert to DataFrame
user_input = pd.DataFrame([all_values], columns=df.columns[:-1])

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Outcome')

place = st.empty()
place.image('https://ludwig.guru/blog/content/images/2017/07/startup-.gif', width=120)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

# Prediction
ans = success_prediction.predict(user_input)[0]

if ans == 1:
    body = f'Startup Successful'
    placeholder.empty()
    place.empty()
    st.success(body)
    progress_bar = st.progress(0)
else:
    body = f'Changes Required to Succeed'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)

st.markdown('Designed by:  Aaryan Bhardwaj & Krishna Goyal')
