import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns= joblib.load('columns.pkl')

st.title("Heart Disease Prediction")

st.markdown("Provode the following details to predict the risk of heart disease:")

#Collecting user input

age= st.slider("Age", 18, 100, 200)
sex= st.selectbox("Sex", ["M","F"])
chest_pain= st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp= st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chloesterol= st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs= st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

#When the user clicks the predict button, we will make a prediction

if st.button("Predict"):

    #raw input dictionary
    input_data = {
        'age': age,
        'Sex_'+ sex :1,
        'RestingBP': resting_bp,
        'Cholesterol': chloesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'ChestPainType_'+ chest_pain: 1,
        'RestingECG_'+ resting_ecg: 1,
        'ExerciseAngina_'+ exercise_angina: 1,
        'ST_Slope_'+ st_slope: 1

    }

    #Creating a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    #Fill in missing columns with zeros
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    #Reorder the columns to match the expected order
    input_df = input_df[expected_columns]

    #Scale the input data
    input_scaled = scaler.transform(input_df)

    #Make prediction
    prediction = model.predict(input_scaled)[0]

    #Display the prediction result
    # Make prediction probability
    # predict_proba returns probabilities for [Class 0 (Low Risk), Class 1 (High Risk)]
    probability_high_risk = model.predict_proba(input_scaled)[0][1]

    # Show result and custom advice based on risk score
    st.subheader(f"Risk Score: {probability_high_risk * 100:.1f}%")

    if probability_high_risk > 0.75:
        st.error("🚨 Critical Risk: You are in the high-risk red zone. Please consult a doctor immediately and avoid intense exertion.")
    elif probability_high_risk > 0.50:
        st.warning("⚠️ Moderate Risk: You are in the elevated risk zone. Moderate daily exercise and diet changes are recommended.")
    elif probability_high_risk > 0.25:
        st.info("ℹ️ Low-to-Moderate Risk: Keep up a healthy lifestyle with regular cardiovascular exercise.")
    else:
        st.success("✅ Low Risk: Excellent! Maintain your current healthy habits and routine checkups.")