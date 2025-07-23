import joblib
import pandas as pd

# تحميل البايبلاين الكامل
pipeline = joblib.load('model_pipeline.pkl')

# بيانات مستخدم جديدة (raw)، نفس الأعمدة الأصلية قبل الـ OneHot
sample = pd.DataFrame([{
    'age': 60,
    'sex': 1,
    'cp': 0,
    'trestbps': 130,
    'chol': 250,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.5,
    'slope': 2,
    'ca': 0,
    'thal': 2
}])

# التنبؤ
prediction = pipeline.predict(sample)
print("🔍 Result:", "Sick" if prediction[0] == 1 else "Fine")
