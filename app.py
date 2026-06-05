import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
import pickle
import os

# get the directory of the current file to load the model and encoders
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# load the trained model and encoders
model = tf.keras.models.load_model(os.path.join(DIR_PATH, 'model.h5'))

with open(os.path.join(DIR_PATH, "gender_mapping.pkl"), "rb") as file:
    gender_mapping = pickle.load(file)

with open(os.path.join(DIR_PATH, "onehot_geography.pkl"), "rb") as file:
    geo_encoder = pickle.load(file)

with open(os.path.join(DIR_PATH, "scaler.pkl"), "rb") as file:
    scaler = pickle.load(file)


# setting up the title and description of the app
st.title("Customer Churn Prediction")
st.write("Enter the customer details to predict if they are likely to churn or not.")

# User Input Fields
geography = st.selectbox("Geography", geo_encoder.categories_[0])
gender = st.selectbox("Gender", list(gender_mapping.keys()))
age = st.slider("Age", 18, 95)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure (Years)", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])

# Preprocess the input data
input_data = pd.DataFrame({
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Balance": [balance],
    "CreditScore": [credit_score],
    "EstimatedSalary": [estimated_salary],
    "Tenure": [tenure],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_cr_card == "Yes" else 0],
    "IsActiveMember": [1 if is_active_member == "Yes" else 0]
})

geography_encoded = geo_encoder.transform(input_data[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(geography_encoded, columns=geo_encoder.get_feature_names_out(["Geography"]))

input_data["Gender"] = input_data["Gender"].map(gender_mapping)
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Remove original Geography column
input_data.drop("Geography", axis=1, inplace=True)

# reordering the input data columns to match the order expected by the scaler and model
input_data = input_data[scaler.feature_names_in_]

# Scaling the input data
input_data_scaled = scaler.transform(input_data)

# Predicting churn
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]


if prediction_probability > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_probability:.2f}.")
else:
    st.write(f"The customer is unlikely to churn with a probability of {prediction_probability:.2f}.")