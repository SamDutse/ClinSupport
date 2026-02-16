import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Triage AI Nigeria", layout="wide")

st.title("🏥 TRIAGE AI NIGERIA")
st.markdown("AI-powered Clinical Decision Support System")

@st.cache_resource
def load_models():
    triage_model = joblib.load("models/triage_model.pkl")
    deterioration_model = joblib.load("models/deterioration_model.pkl")
    readmission_model = joblib.load("models/readmission_model.pkl")
    return triage_model, deterioration_model, readmission_model

triage_model, deterioration_model, readmission_model = load_models()

# def explain_prediction(model, input_df):
#     final_model = model.named_steps["classifier"]
#     transformed_input = model[:-1].transform(input_df)
#     explainer = shap.Explainer(final_model)
#     shap_values = explainer(transformed_input)

#     st.subheader("🔎 SHAP Explanation")
#     fig, ax = plt.subplots()
#     shap.plots.waterfall(shap_values[0], show=False)
#     st.pyplot(fig)

model_option = st.sidebar.selectbox(
    "Select Model",
    ["Triage Prediction", "Deterioration Prediction", "Readmission Prediction"]
)

# =====================================================
# TRIAGE MODEL
# =====================================================
if model_option == "Triage Prediction":

    st.header("🩺 Emergency Triage Risk")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 45)
        heart_rate = st.number_input("Heart Rate", 40, 200, 90)
        systolic_bp = st.number_input("Systolic BP", 70, 250, 120)
        diastolic_bp = st.number_input("Diastolic BP", 40, 150, 80)
        temperature = st.number_input("Temperature (°C)", 34.0, 42.0, 37.0)
        respiratory_rate = st.number_input("Respiratory Rate", 10, 40, 20)
        spo2 = st.number_input("SpO2 (%)", 50, 100, 98)

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        trauma = st.selectbox("Trauma", [0, 1])
        difficulty_breathing = st.selectbox("Difficulty Breathing", [0, 1])
        seizure = st.selectbox("Seizure", [0, 1])
        chest_pain = st.selectbox("Chest Pain", [0, 1])
        pregnant = st.selectbox("Pregnant", [0, 1])
        fever = st.selectbox("Fever", [0, 1])

    if st.button("Predict Triage Risk"):

        input_df = pd.DataFrame([{
            "age": age,
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "temperature": temperature,
            "respiratory_rate": respiratory_rate,
            "spo2": spo2,
            "sex": sex,
            "trauma": trauma,
            "difficulty_breathing": difficulty_breathing,
            "seizure": seizure,
            "chest_pain": chest_pain,
            "pregnant": pregnant,
            "fever": fever
        }])

        prediction = triage_model.predict(input_df)[0]
        probability = triage_model.predict_proba(input_df)[0][1]

        st.success(f"Prediction: {prediction}")
        st.write(f"Risk Probability: {probability:.2f}")

        # explain_prediction(triage_model, input_df)


# =====================================================
# DETERIORATION MODEL
# =====================================================
elif model_option == "Deterioration Prediction":

    st.header("⚠️ Clinical Deterioration Risk")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 60)
        heart_rate = st.number_input("Heart Rate", 40, 200, 95)
        respiratory_rate = st.number_input("Respiratory Rate", 10, 40, 20)
        systolic_bp = st.number_input("Systolic BP", 70, 250, 120)
        temperature = st.number_input("Temperature (°C)", 34.0, 42.0, 37.0)
        spo2 = st.number_input("SpO2 (%)", 50, 100, 98)

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        ward_type = st.selectbox("Ward Type", ["General", "ICU", "Emergency"])
        hypertension = st.selectbox("Hypertension", [0, 1])
        asthma = st.selectbox("Asthma", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])
        creatinine = st.number_input("Creatinine", 0.1, 10.0, 1.0)
        wbc = st.number_input("WBC", 1000, 50000, 8000)
        blood_glucose = st.number_input("Blood Glucose", 50, 500, 120)
        hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.0)

    if st.button("Predict Deterioration Risk"):

        input_df = pd.DataFrame([{
            "age": age,
            "heart_rate": heart_rate,
            "respiratory_rate": respiratory_rate,
            "systolic_bp": systolic_bp,
            "temperature": temperature,
            "spo2": spo2,
            "sex": sex,
            "ward_type": ward_type,
            "hypertension": hypertension,
            "asthma": asthma,
            "diabetes": diabetes,
            "creatinine": creatinine,
            "wbc": wbc,
            "blood_glucose": blood_glucose,
            "hemoglobin": hemoglobin
        }])

        prediction = deterioration_model.predict(input_df)[0]
        probability = deterioration_model.predict_proba(input_df)[0][1]

        st.success(f"Prediction: {prediction}")
        st.write(f"Risk Probability: {probability:.2f}")

        # explain_prediction(deterioration_model, input_df)


# =====================================================
# READMISSION MODEL
# =====================================================
elif model_option == "Readmission Prediction":

    st.header("🔁 30-Day Readmission Risk")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 55)
        length_of_stay = st.number_input("Length of Stay", 1, 60, 5)
        previous_admissions = st.number_input("Previous Admissions", 0, 20, 1)
        comorbidity_count = st.number_input("Comorbidity Count", 0, 10, 2)
        discharge_medication_count = st.number_input("Discharge Medication Count", 0, 20, 3)

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        primary_diagnosis = st.selectbox(
            "Primary Diagnosis",
            ["Hypertension", "Diabetes", "Pneumonia", "Heart Failure"]
        )
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        followup_scheduled = st.selectbox("Follow-up Scheduled", [0, 1])

    if st.button("Predict Readmission Risk"):

        input_df = pd.DataFrame([{
            "age": age,
            "length_of_stay": length_of_stay,
            "previous_admissions": previous_admissions,
            "comorbidity_count": comorbidity_count,
            "discharge_medication_count": discharge_medication_count,
            "sex": sex,
            "primary_diagnosis": primary_diagnosis,
            "residence": residence,
            "followup_scheduled": followup_scheduled
        }])

        prediction = readmission_model.predict(input_df)[0]
        probability = readmission_model.predict_proba(input_df)[0][1]

        st.success(f"Prediction: {prediction}")
        st.write(f"Risk Probability: {probability:.2f}")

        # explain_prediction(readmission_model, input_df)
