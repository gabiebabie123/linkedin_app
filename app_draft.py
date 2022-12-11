import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go 
from PIL import Image 

photo = Image.open('Photo.png')
st.image(photo, caption = "Image taken from Google")


#Income Below
income = st.selectbox("What is your income range?",
    options = ["Less than $10,000",
              "10,000 to under $20,000",
              "20,000 to under $30,000",
              "30,000 to under $40,000",
              "40,000 to under $50,000",
              "50,000 to under $75,000",
              "75,000 to under $100,000",
              "100,000 to under $150,000",
              "150,000 or above?"
               ])


st.write (f"Income Selected: {income}")

if income == "Less than $10,000":
    income = 1
elif income == "10,000 to under $20,000":
    income = 2
elif income == "20,000 to under $30,000":
    income = 3
elif income == "30,000 to under $40,000":
    income = 4
elif income == "40,000 to under $50,000":
    income = 5
elif income == "50,000 to under $75,000":
    income = 6
elif income == "75,000 to under $100,000":
    income = 7
elif income == "100,000 to under $150,000":
    income = 8
else:
    income = 9

#Education Below
education = st.selectbox("Highest Education Level Achieved",
    options = ["Less than High School (Grades 1-8 or no formal schooling",
              "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
              "High school graduate (Grade 12 with diploma or GED certificate)",
              "Some college, no degree (includes some community college)",
              "Two-year associate degree from a college or university",
              "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
              "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
              "Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD)"
               ])


st.write (f"Education Level Selected: {education}")

if education == "Less than High School (Grades 1-8 or no formal schooling":
    education = 1
elif education == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    education = 2
elif education == "High school graduate (Grade 12 with diploma or GED certificate)":
    education = 3
elif education == "Some college, no degree (includes some community college)":
    education = 4
elif education == "Two-year associate degree from a college or university":
    education = 5
elif education == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education = 7
elif education == "Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD)":
    education = 8

#Parental Status

parent = st.selectbox("Are you a parent of a child under 18 living in your home?",
    options = ["Yes",
              "No"])

st.write (f"Parental Status Selected: {parent}")

if parent == "Yes":
    parent = 1
else:
    parent = 0

#Marital Status

marital = st.selectbox("What is you current marital status?",
    options = ["Married",
                "Single/Complicated"])

st.write (f"Marital Status Selected: {marital}")
if marital == "Married":
    marital = 1
else:
    marital = 0

#Gender
gender = st.selectbox("What is your gender?",
    options = ["Male",
                "Female"])

st.write (f"Gender Selected: {gender}")
if gender == "Male":
    gender = 0
else:
    gender = 1

#Age 
age = st.number_input("What is your age?",
    min_value = 1,
    max_value = 99,
    value = 25)
st.write("Age given =", age)

#INSERTTING CODE FROM QUESTIONS#
s = pd.read_csv("C:/Users/mende/OneDrive/Georgetown/Programming 2/social_media_usage.csv")

#Clean Function
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

#SS Dataframe
ss = pd.DataFrame({
    "sm_li": np.where(s["web1h"] == 1, 1, 0),
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
})

ss = ss.dropna()

y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]


x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify = y,    #same number of target in training & test set
                                                   test_size = 0.2, #hold out 20% of data for testing
                                                   random_state = 420) #set for reproducibility

lr = LogisticRegression(class_weight = "balanced")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

newdata = pd.DataFrame({
    "income": [income],
    "education": [education],
    "parent": [parent],
    "married": [marital],
    "female": [gender],
    "age": [age]
})

user_prob = (lr.predict_proba(newdata))[0][1]
user_prob = (round(user_prob, 4))
st.markdown ({user_prob})


#Color Wheel

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = user_prob,
    title = {'text': f"Probability that you are a LinkedIn user"},
    gauge = {"axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, .44], "color":"red"},
                {"range": [.45, .55], "color":"yellow"},
                {"range": [.56, 1], "color":"green"}
            ],
            "bar":{"color":"black"}}
))

st.plotly_chart(fig)



  
