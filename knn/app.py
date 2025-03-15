import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import sklearn

# Cek versi scikit-learn
print("scikit-learn version:", sklearn.__version__)

# Load preprocessed data
df_avg = pd.read_csv("preprocessed_data.csv", index_col=0)

# Pastikan hanya mengambil fitur yang sesuai dengan model
feature_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                   "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

diabetic_avg = df_avg.loc["Diabetic", feature_columns].values.reshape(1, -1)
non_diabetic_avg = df_avg.loc["Non-Diabetic", feature_columns].values.reshape(1, -1)


# Load trained model
model = load("knn_diabetes_model.joblib")

# ---- UI Styling ----
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.markdown("""
    <style>
        .big-font { font-size:25px !important; }
        .prediction-box { 
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            background-color: #1E1E1E;
            color: white;
        }
        .positive { background-color: #FF4B4B !important; }
        .negative { background-color: #4CAF50 !important; }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.title("🔍 Diabetes Prediction Dashboard")
st.write("Masukkan data pasien untuk mengetahui prediksi diabetes.")

# ---- Sidebar Input ----
st.sidebar.header("📝 Masukkan Data Pasien")
col1, col2 = st.sidebar.columns(2)

pregnancies = col1.number_input("Pregnancies", 0, 20, 1)
glucose = col2.number_input("Glucose", 0, 200, 100)
blood_pressure = col1.number_input("Blood Pressure", 0, 130, 70)
skin_thickness = col2.number_input("Skin Thickness", 0, 100, 20)
insulin = col1.number_input("Insulin", 0, 900, 80)
bmi = col2.number_input("BMI", 0.0, 70.0, 25.0)
dpf = col1.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = col2.number_input("Age", 0, 120, 30)

# ---- Predict Button ----
if st.sidebar.button("🔮 Predict", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

    color_class = "positive" if result == "Diabetic" else "negative"
    st.markdown(f'<div class="prediction-box {color_class}">{result} ({confidence*100:.2f}% confidence)</div>', unsafe_allow_html=True)

# ---- Prediksi Rata-rata Pasien ----
st.sidebar.subheader("🔍 Cek Prediksi Rata-rata Pasien")

if st.sidebar.button("🔬 Prediksi Rata-rata Diabetes", use_container_width=True):
    pred_avg_diabetic = model.predict(diabetic_avg)
    proba_avg_diabetic = model.predict_proba(diabetic_avg)
    
    result_avg_diabetic = "Diabetic" if pred_avg_diabetic[0] == 1 else "Not Diabetic"
    confidence_avg_diabetic = proba_avg_diabetic[0][1] if pred_avg_diabetic[0] == 1 else proba_avg_diabetic[0][0]

    color_class = "positive" if result_avg_diabetic == "Diabetic" else "negative"
    st.markdown(f'<div class="prediction-box {color_class}">Rata-rata Pasien Diabetes: {result_avg_diabetic} ({confidence_avg_diabetic*100:.2f}% confidence)</div>', unsafe_allow_html=True)

if st.sidebar.button("🔬 Prediksi Rata-rata Non-Diabetes", use_container_width=True):
    pred_avg_non_diabetic = model.predict(non_diabetic_avg)
    proba_avg_non_diabetic = model.predict_proba(non_diabetic_avg)
    
    result_avg_non_diabetic = "Diabetic" if pred_avg_non_diabetic[0] == 1 else "Not Diabetic"
    confidence_avg_non_diabetic = proba_avg_non_diabetic[0][1] if pred_avg_non_diabetic[0] == 1 else proba_avg_non_diabetic[0][0]

    color_class = "positive" if result_avg_non_diabetic == "Diabetic" else "negative"
    st.markdown(f'<div class="prediction-box {color_class}">Rata-rata Pasien Non-Diabetes: {result_avg_non_diabetic} ({confidence_avg_non_diabetic*100:.2f}% confidence)</div>', unsafe_allow_html=True)

# ---- Data Visualization ----
st.subheader("📊 Distribusi Data Diabetes")
df = pd.read_csv("diabetes.csv")  # Pastikan dataset ada
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df['Glucose'], kde=True, bins=30, ax=ax, color="#FF4B4B")
ax.set_title("Distribusi Glukosa dalam Dataset", fontsize=14)
ax.set_xlabel("Glucose Level")
ax.set_ylabel("Count")
st.pyplot(fig)


df_avg = pd.read_csv("preprocessed_data.csv", index_col=0)
diabetic_avg = df_avg.loc["Diabetic"].to_dict()
non_diabetic_avg = df_avg.loc["Non-Diabetic"].to_dict()
