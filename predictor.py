import streamlit as st
import numpy as np
import joblib

st.set_page_config(layout="wide")

scaler = joblib.load("Scaler.pkl")

#streamlit setting
st.title("Restaurant Rating Predictor")
st.caption("Predict the ratings")
st.divider()
averagecost = st.number_input("Enter the average cost for two", min_value=50,max_value=999999,value=1000, step=200)

tablebooking = st.selectbox("Does the restaurant has table booking?", ["Yes","No"])
bookingstatus = 1 if tablebooking == "Yes" else 0

onlinedelivery = st.selectbox("Does the restaurant has online delivery?", ["Yes","No"])
deliverystatus = 1 if onlinedelivery == "Yes" else 0

pricerange = st.selectbox("What is the price range?(1 Cheapest, 4 Expensive)", ["1","2","3","4"])
pricerange = int(pricerange)

button = st.button("Predict the Review")
st.divider()
model = joblib.load("mlmodel.pkl")

values = [[averagecost, bookingstatus, deliverystatus, pricerange]] 
my_X_values = np.array(values)

X = scaler.transform(my_X_values) 

if button:
    prediction = model.predict(X)
    st.write(prediction)

#Above 2 and below 2.5 is Poor
#Above 2.5 and below 3.5 is Average
#Above 3.5 and below 4 is Good
#Above 4 and below 4.5 is Very Good
#Above 4.5 is Excellent

    if prediction < 2.5 :
        st.write("Poor")
    elif prediction < 3.5 :
        st.write("Average")
    elif prediction < 4 :
        st.write("Good")
    elif prediction < 4.5 :
        st.write("Very Good")
    else:
        st.write("Excellent")        
