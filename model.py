'''
my proj


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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from sklearn.pipeline import Pipeline


# loading data
heart_disease = fetch_ucirepo(id=45) 

#  DataFrame
df = heart_disease.data.original
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns='num', inplace=True)

# Exclude target from features
df_features = df.drop(columns='target')

# Separate numeric and categorical columns
numeric_features = df_features.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df_features.select_dtypes(include=['object']).columns

# Define preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Build column transformer (excluding target)
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fit preprocessor on training data
preprocessor.fit(df.drop(columns='target'))

# Transform the data
X_processed = preprocessor.fit_transform(df.drop(columns='target'))
#########################################################################

# --- 1. EDA ---
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# --- 2. PCA ---
df_numeric = df.select_dtypes(include=['int64', 'float64'])
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df_numeric)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_imputed)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Variance Retained")
plt.grid(True)
plt.show()
###############################################################

# --- Feature Selection ---
target_col = 'target' if 'target' in df.columns else df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# --- 1. Feature Importance باستخدام Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X.select_dtypes(include=['int64', 'float64']), y)

importances = rf.feature_importances_
feature_names = X.select_dtypes(include=['int64', 'float64']).columns

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

X_cat = X.select_dtypes(include=['int64']).copy()
X_cat = X_cat.fillna(0)  

chi2_selector = SelectKBest(chi2, k=5)
chi2_selector.fit(X_cat, y)

chi2_features = X_cat.columns[chi2_selector.get_support()]
print("Chi-Square:", list(chi2_features))
###################################################

# --- Supervised Learning ---
df_clean = df.copy()
imputer = SimpleImputer(strategy='most_frequent')
df_clean[df_clean.columns] = imputer.fit_transform(df_clean)

X = df_clean.drop(columns='target')
y = df_clean['target']

# OneHot
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n📌 Model: {name}")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

    #  ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()
#####################################################################

# --- Unsupervised Learning ---
X_cluster = df_clean.drop(columns='target')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method - KMeans')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_scaled)

df_clean['kmeans_cluster'] = clusters_kmeans

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

hc = AgglomerativeClustering(n_clusters=2)
clusters_hc = hc.fit_predict(X_scaled)
df_clean['hc_cluster'] = clusters_hc

print("\n📊 Hierarchical Cluster vs Actual Target:\n")
print(pd.crosstab(df_clean['hc_cluster'], df_clean['target']))
#############################################################################

# --- Hyperparameter Tuning ---
model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# ---  Randomized Search ---
random_search = RandomizedSearchCV(model, param_distributions=param_grid,
                                   n_iter=10, cv=5, scoring='f1', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("\n✅ best parameters")
print(random_search.best_params_)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n📊 Evaluation of Tuned Model:\n")
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

###################################################################

# --- Exporting Model ---
joblib.dump(best_model, 'final_model.pkl')
print("✅ Model saved as final_model.pkl")

# Refit the preprocessor and save full pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

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
