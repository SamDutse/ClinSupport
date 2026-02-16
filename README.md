# 🏥 TRIAGE AI NIGERIA

AI-Powered Clinical Decision Support System for Nigerian Hospitals.

Triage AI Nigeria is a machine learning–based hospital risk prediction system designed to support frontline healthcare workers in resource-constrained environments.

The system provides:

- 🩺 Emergency Triage Risk Prediction  
- ⚠️ Clinical Deterioration Risk Prediction  
- 🔁 30-Day Readmission Risk Prediction  

Built using synthetic hospital data for initial development, with architecture designed to seamlessly transition to real hospital patient records.

---

# 🚀 Live Application

🔗 **Access the deployed system here:**

👉 https://clinsupport-jkvjjmhxtfbhaxw5nwddaa.streamlit.app

*(Replace the link above with your Streamlit Cloud deployment URL after deployment.)*

---

## 🧠 Project Overview

Nigeria’s healthcare infrastructure faces:

- Overcrowded emergency departments
- Limited triage standardization
- Late detection of patient deterioration
- High 30-day readmission rates

This system uses machine learning pipelines to assist clinicians in making faster and more data-driven decisions.

---

## 🔍 Models Included

### 1️⃣ Triage Prediction
Predicts emergency triage risk level based on:
- Vital signs
- Symptoms
- Trauma indicators
- Pregnancy status
- Fever indicators

### 2️⃣ Clinical Deterioration Prediction
Predicts likelihood of in-hospital deterioration using:
- Vital signs
- Lab results
- Comorbidities
- Ward type

### 3️⃣ 30-Day Readmission Prediction
Predicts readmission risk using:
- Length of stay
- Comorbidity count
- Diagnosis
- Discharge medication count
- Follow-up scheduling
- Residence type

---

## 🏗️ Tech Stack

- Python
- Scikit-learn (Pipelines + ColumnTransformers)
- Pandas
- Streamlit
- Joblib
- SHAP (optional, currently disabled in production version)

---

## 📁 Project Structure

triage_ai_nigeria/
│
├── app.py
├── triage_model.ipynb
├── deterioration_model.ipynb
├── readmission_model.ipynb
├── models/
│ ├── triage_model.pkl
│ ├── deterioration_model.pkl
│ └── readmission_model.pkl
└── data/


---

## ⚙️ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/yourusername/triage_ai_nigeria.git
cd triage_ai_nigeria
