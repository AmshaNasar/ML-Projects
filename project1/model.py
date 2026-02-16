import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load model and feature order

model = joblib.load("best_model.pkl")
feature_order = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Gender",
    "Extracurricular_Activities"
]

st.title("Student Exam Score Predictor")

# Collect user inputs
hours_studied = st.slider("Hours Studied", 0, 40, 12)
attendance = st.slider("Attendance (%)", 0, 100, 80)
sleep_hours = st.slider("Sleep Hours per Night", 0, 12, 7)
previous_scores = st.slider("Previous Scores (%)", 0, 100, 70)
tutoring_sessions = st.slider("Tutoring Sessions per Week", 0, 5, 2)
physical_activity = st.slider("Physical Activity (hours/week)", 0, 5, 3)
gender = st.selectbox("Gender", ["Male", "Female"])
extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

# Encode categorical features the same way as training
gender_encoded = 1 if gender == "Male" else 0
extra_encoded = 1 if extracurricular == "Yes" else 0

# Build input data as a DataFrame with correct column streamliorder
input_dict = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Tutoring_Sessions": tutoring_sessions,
    "Physical_Activity": physical_activity,
    "Gender": gender_encoded,
    "Extracurricular_Activities": extra_encoded
}

input_df = pd.DataFrame([input_dict])[feature_order]

if st.button("Predict Exam Score"):
    prediction = model.predict(input_df)[0]

    # Clamp score between 0–100
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Exam Score: {prediction:.2f}%")

    # Debug: Show what the model actually sees
    st.write("Input data passed to model:", input_df)

