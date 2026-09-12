import streamlit as st
import pandas as pd
import pickle

# Load model
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Risk Prediction")
st.write("Enter patient details below.")

age = st.number_input("Age", 1, 120, 50)

gender = st.selectbox(
    "Gender",
    [1, 2],
    format_func=lambda x: "Female" if x == 1 else "Male"
)

height = st.number_input("Height (cm)", 50.0, 250.0, 170.0)

weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0)

ap_hi = st.number_input("Systolic Blood Pressure", 50, 250, 120)

ap_lo = st.number_input("Diastolic Blood Pressure", 30, 150, 80)

cholesterol = st.selectbox(
    "Cholesterol",
    [1, 2, 3],
    format_func=lambda x: {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }[x]
)

gluc = st.selectbox(
    "Glucose",
    [1, 2, 3],
    format_func=lambda x: {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }[x]
)

smoke = st.selectbox(
    "Smoking",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

alco = st.selectbox(
    "Alcohol Intake",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

active = st.selectbox(
    "Physically Active",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

if st.button("Predict"):

    sample = pd.DataFrame([[
        age,
        gender,
        height,
        weight,
        ap_hi,
        ap_lo,
        cholesterol,
        gluc,
        smoke,
        alco,
        active
    ]], columns=[
        "age",
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active"
    ])

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Higher predicted risk")
    else:
        st.success("Lower predicted risk")

    st.write(f"Class 1 probability: {probability * 100:.2f}%")