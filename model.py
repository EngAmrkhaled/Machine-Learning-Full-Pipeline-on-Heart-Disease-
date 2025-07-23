'''
my proj


'''
'''
import numpy  as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly._subplots import make_subplots

import ucimlrepo
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import ucimlrepo
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from sklearn.pipeline import Pipeline


# تحميل بيانات Heart Disease من UCI
heart_disease = fetch_ucirepo(id=45) 

# استخراج الـ DataFrame
df = heart_disease.data.original
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns='num', inplace=True)


# 2. فصل البيانات لأعمدة رقمية ونصية
# استبعاد target من الأعمدة
df_features = df.drop(columns='target')

# فصل الأعمدة الرقمية والنصية بدون target
numeric_features = df_features.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df_features.select_dtypes(include=['object']).columns


# إنشاء pipeline للمعالجة
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# بناء preprocessor بدون target
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# تدريب الـ preprocessor على بيانات التدريب فقط (بدون target)
preprocessor.fit(df.drop(columns='target'))


# 5. تنفيذ المعالجة
X_processed = preprocessor.fit_transform(df.drop(columns='target'))
#########################################################################

# --- 1. EDA ---
# رسم التوزيع لكل عمود رقمي
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# خريطة الارتباط بين المتغيرات
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# --- 2. PCA ---
# إزالة الأعمدة النصية إن وجدت
# فقط الأعمدة الرقمية
df_numeric = df.select_dtypes(include=['int64', 'float64'])

# معالجة القيم الناقصة
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df_numeric)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_imputed)



# تنفيذ PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# رسم نسبة التباين المحتفظ بها
plt.figure(figsize=(8,5))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Variance Retained")
plt.grid(True)
plt.show()
###############################################################
# بنفترض إن المتغير الهدف اسمه 'target' (غيّر الاسم لو مختلف)
target_col = 'target' if 'target' in df.columns else df.columns[-1]

# نجهز البيانات (من غير OneHot هنا - علشان Chi-Square)
X = df.drop(columns=[target_col])
y = df[target_col]

# --- 1. Feature Importance باستخدام Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X.select_dtypes(include=['int64', 'float64']), y)

importances = rf.feature_importances_
feature_names = X.select_dtypes(include=['int64', 'float64']).columns

# عرض النتائج
feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(8,6))
feat_importance.plot(kind='bar')
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance")
plt.grid(True)
plt.show()

# --- 2. Recursive Feature Elimination (RFE) ---
rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
rfe.fit(X.select_dtypes(include=['int64', 'float64']), y)

selected_features_rfe = feature_names[rfe.support_]
print(" RFE:", list(selected_features_rfe))

# --- 3. Chi-Square Test ---
# نجهز بيانات تصنيفية + هدف
X_cat = X.select_dtypes(include=['int64']).copy()
X_cat = X_cat.fillna(0)  # نتأكد مفيش قيم ناقصة

chi2_selector = SelectKBest(chi2, k=5)
chi2_selector.fit(X_cat, y)

chi2_features = X_cat.columns[chi2_selector.get_support()]
print("Chi-Square:", list(chi2_features))
###################################################
df_clean = df.copy()
imputer = SimpleImputer(strategy='most_frequent')
df_clean[df_clean.columns] = imputer.fit_transform(df_clean)

# --- البيانات ---
X = df_clean.drop(columns='target')
y = df_clean['target']

# لو في بيانات نصية، نحولها OneHot
X = pd.get_dummies(X, drop_first=True)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- النماذج ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

# --- تدريب وتقييم ---
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n📌 Model: {name}")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

    # رسم ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})")

# رسم ROC Curve لكل النماذج
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()
#####################################################################

# نستخدم النسخة المعالجة
X_cluster = df_clean.drop(columns='target')

# توحيد البيانات
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# تجربة أكثر من K
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# رسم Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method - KMeans')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# نضيف النتيجة للبيانات
df_clean['kmeans_cluster'] = clusters_kmeans

# مقارنة الكلاستر بالـ target
print("\n📊 K-Means Cluster vs Actual Target:\n")
print(pd.crosstab(df_clean['kmeans_cluster'], df_clean['target']))


linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# تنفيذ الكلاستر
hc = AgglomerativeClustering(n_clusters=2)
clusters_hc = hc.fit_predict(X_scaled)
df_clean['hc_cluster'] = clusters_hc

# مقارنة الكلاستر بـ target
print("\n📊 Hierarchical Cluster vs Actual Target:\n")
print(pd.crosstab(df_clean['hc_cluster'], df_clean['target']))
#############################################################################

# --- تحديد موديل ---
model = RandomForestClassifier(random_state=42)

# --- تحديد الشبكة ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# --- تنفيذ Randomized Search ---
random_search = RandomizedSearchCV(model, param_distributions=param_grid,
                                   n_iter=10, cv=5, scoring='f1', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# --- أفضل موديل ---
best_model = random_search.best_estimator_

print("\n✅ best parameters")
print(random_search.best_params_)

# --- تقييم الأداء على test set ---

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n📊 Evaluation of Tuned Model:\n")
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

###################################################################

# حفظ أفضل موديل بعد التوليف
joblib.dump(best_model, 'final_model.pkl')
print("✅ Model saved as final_model.pkl")

# إنشاء preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 👇 هنا تعمل fit بدون target
preprocessor.fit(df.drop(columns='target'))

full_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', best_model)
])
joblib.dump(full_pipeline, 'model_pipeline.pkl')
print("✅ Full pipeline saved as model_pipeline.pkl")

'''
# تحميل النموذج المحفوظ
loaded_model = joblib.load('final_model.pkl')

# استخدامه في التنبؤ
prediction = loaded_model.predict(X_test)
'''