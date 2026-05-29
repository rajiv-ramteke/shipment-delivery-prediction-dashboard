
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Shipment Prediction", layout="wide")

st.title("Shipment Delivery Prediction Dashboard")

# Load model
model = joblib.load("shipment_model.pkl")

# Sidebar
st.sidebar.header("Shipment Details")

warehouse = st.sidebar.selectbox(
    "Warehouse Block",
    [0,1,2,3,4]
)

shipment = st.sidebar.selectbox(
    "Mode of Shipment",
    [0,1,2]
)

care_calls = st.sidebar.slider(
    "Customer Care Calls",
    1,
    10,
    4
)

rating = st.sidebar.slider(
    "Customer Rating",
    1,
    5,
    3
)

cost = st.sidebar.number_input(
    "Cost of Product",
    50,
    500,
    200
)

prior = st.sidebar.slider(
    "Prior Purchases",
    1,
    10,
    3
)

importance = st.sidebar.selectbox(
    "Product Importance",
    [0,1,2]
)

gender = st.sidebar.selectbox(
    "Gender",
    [0,1]
)

discount = st.sidebar.slider(
    "Discount Offered",
    0,
    100,
    10
)

weight = st.sidebar.number_input(
    "Weight in gms",
    100,
    6000,
    2000
)

# Input Data
input_data = np.array([[
    warehouse,
    shipment,
    care_calls,
    rating,
    cost,
    prior,
    importance,
    gender,
    discount,
    weight
]])

# Prediction
if st.button("Predict Shipment Status"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Shipment Will Be Delayed")
    else:
        st.success("Shipment Will Reach On Time")

# Dashboard Charts
st.subheader("Analytics Dashboard")

df = pd.read_csv("Train.csv")

st.bar_chart(df['Customer_rating'])

st.line_chart(df['Cost_of_the_Product'])

st.area_chart(df['Discount_offered'])

# Distribution
st.subheader("Shipment Distribution")

st.write(
    df['Reached.on.Time_Y.N'].value_counts()
)

# Raw Data
if st.checkbox("Show Dataset"):
    st.write(df)
