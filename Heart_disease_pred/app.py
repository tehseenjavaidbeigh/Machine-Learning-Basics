import streamlit as st
import pandas as pd
import joblib

model = joblib.load('Logistic Regression.pkl')
scaler = joblib.load('scaler_heart.pkl')
columns = joblib.load('columns_heart.pkl')

st.title("Heart disease prediction using Logistic Regression")
st.markdown("Provide the following details:")

age = st.slider("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", ['Normal', 'ST', 'LVH'])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'Sex': 'M' if sex == "Male" else 'F',
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    input_df = pd.DataFrame([raw_input])
    input_df = pd.get_dummies(input_df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]
    input_df_scaled = scaler.transform(input_df)
    prediction = model.predict(input_df_scaled)[0]

    if prediction == 1:
        st.error("The model predicts that you have heart disease.")
    else:
        st.success("The model predicts that you do not have heart disease.")