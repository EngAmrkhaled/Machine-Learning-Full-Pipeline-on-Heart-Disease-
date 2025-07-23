import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model_pipeline.pkl')

# Page title
st.title("🔬 Heart Disease Prediction App")
st.markdown("Enter your health information to check the likelihood of heart disease.")

# User inputs
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 130)
chol = st.slider("Cholesterol (chol)", 100, 600, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert to DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])

# Predict when button is clicked
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result:")
    st.write("✅ No heart disease risk" if prediction == 0 else "⚠️ Possible heart disease risk")

    st.subheader("📊 Prediction Probability:")
    st.write(f"{prob*100:.2f} %")
