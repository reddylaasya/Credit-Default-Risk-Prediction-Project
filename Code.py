"""
Credit Default Risk Prediction Project
Author: Laasya Reddy Dendi

In this project, we are using LendingClub loan data to create two models:
1. A classification model to predict the probability of loan default.
2. A regression model to estimate the potential default amount.
3. We then calculate the Expected Loss = P(Default) × Default Amount.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, classification_report
)

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from lightgbm import LGBMClassifier, LGBMRegressor

# Handle class imbalance
from imblearn.over_sampling import SMOTE

# Clustering methods
from sklearn.cluster import KMeans, DBSCAN

# Clean output
warnings.filterwarnings("ignore")

# ------------------- Data Loading -------------------

print("Loading LendingClub dataset...")

# Load dataset (we are sampling 100,000 rows to save memory)
df = pd.read_csv("/Users/laasyareddydendi/Downloads/archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv", low_memory=False)

# Random sampling - set seed for reproducibility
df= df.sample(n=100000, random_state=42)

# Keep only columns we actually need
columns = [
    'loan_amnt', 'term', 'emp_length', 'home_ownership', 'annual_inc',
    'purpose', 'dti', 'fico_range_high', 'fico_range_low', 'loan_status', 'revol_util'
]
df = df[columns]

# ------------------- Label Engineering -------------------

# Define which statuses are considered as "default"
default_labels = ['Charged Off', 'Default', 'Late (31-120 days)', 'Does not meet the credit policy. Status:Charged Off']
df = df[df['loan_status'].notna()]

# Apply our labeling (1 for default, 0 for non-default)
df['default'] = df['loan_status'].apply(lambda x: 1 if x in default_labels else 0)

# Cleaning up interest rate column by stripping off the '%' symbol
df['revol_util'] = pd.to_numeric(df['revol_util'].astype(str).str.rstrip('%'), errors='coerce')

# ------------------- Preprocessing -------------------

# Handle missing values in key numeric fields
imputer = SimpleImputer(strategy='mean')
num_cols = ['annual_inc', 'dti', 'fico_range_high', 'fico_range_low', 'revol_util']
df[num_cols] = imputer.fit_transform(df[num_cols])

# Converting 'term' and 'emp_length' to numerical formats — prepping them for the model's eyes
df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').fillna(0).astype(float)

# Encoding categorical features using LabelEncoder — giving machines what they understand
df['home_ownership'] = LabelEncoder().fit_transform(df['home_ownership'])
df['purpose'] = LabelEncoder().fit_transform(df['purpose'])

# FICO score midpoint + log annual income = features that often carry predictive power in finance
df['fico_score'] = (df['fico_range_high'] + df['fico_range_low']) / 2
df['log_annual_inc'] = np.log1p(df['annual_inc'])

# ------------------- Features and Targets -------------------

# Define the features and target labels
features = [
    'loan_amnt', 'term', 'emp_length', 'home_ownership', 'log_annual_inc',
    'purpose', 'dti', 'revol_util', 'fico_score'
]

X = df[features]
y_default = df['default']
y_loss = df['loan_amnt'] * df['default']  # 0 if no default

# Train-test split
X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
    X, y_default, y_loss, test_size=0.2, random_state=42
)

# Scaling - important for logistic regression, giving all features equal footing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------- Classification -------------------

# Using SMOTE to rebalance my training data
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_cls_bal, y_cls_bal = smote.fit_resample(X_train_scaled, y_train_cls)

# Train LightGBM Classifier - handling non-linear patterns efficiently
clf_lgb = LGBMClassifier(n_estimators=50, max_depth=8, random_state=42)
clf_lgb.fit(X_cls_bal, y_cls_bal)

# Predictions
pred_cls = clf_lgb.predict(X_test_scaled)
prob_cls = clf_lgb.predict_proba(X_test_scaled)[:, 1]

# Print classification model results
print("\n--- LightGBM Classifier ---")
print("Accuracy:", accuracy_score(y_test_cls, pred_cls))
print("F1 Score:", f1_score(y_test_cls, pred_cls))
print("AUC-ROC:", roc_auc_score(y_test_cls, prob_cls))
print(classification_report(y_test_cls, pred_cls))

# Train Logistic Regression (as a baseline model)
clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_cls_bal, y_cls_bal)

# Predictions for logistic regression
pred_lr = clf_lr.predict(X_test_scaled)
prob_lr = clf_lr.predict_proba(X_test_scaled)[:, 1]

# Print logistic regression results
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test_cls, pred_lr))
print("F1 Score:", f1_score(y_test_cls, pred_lr))
print("AUC-ROC:", roc_auc_score(y_test_cls, prob_lr))

# ------------------- Regression -------------------

# LightGBM again, but now for estimating the financial hit — predicting potential default amount
reg_lgb = LGBMRegressor(n_estimators=50, max_depth=8, random_state=42)
reg_lgb.fit(X_train_scaled, y_train_reg)
pred_reg_lgb = reg_lgb.predict(X_test_scaled)

# Print regression model results
print("\n--- LGBM Regressor ---")
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred_reg_lgb)))
print("R²:", r2_score(y_test_reg, pred_reg_lgb))

# Train Linear Regression (as a baseline model)
reg_lin = LinearRegression()
reg_lin.fit(X_train_scaled, y_train_reg)
pred_reg_lin = reg_lin.predict(X_test_scaled)

# Print linear regression results
print("\n--- Linear Regression ---")
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred_reg_lin)))
print("R²:", r2_score(y_test_reg, pred_reg_lin))

# ------------------- Expected Loss -------------------

# Combining classification and regression outputs to estimate Expected Loss
expected_loss = prob_cls * pred_reg_lgb
df_result = pd.DataFrame({
    "P_Default": prob_cls,
    "Predicted_Loss": pred_reg_lgb,
    "Expected_Loss": expected_loss
})

# top predictions
print("\n--- Sample Expected Loss ---")
print(df_result.head())

# ------------------- Clustering -------------------

# Clustering with KMeans and DBSCAN
cluster_feats = ['loan_amnt', 'log_annual_inc', 'fico_score', 'dti', 'revol_util']
X_cluster = scaler.fit_transform(df[cluster_feats])

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_cluster)

# DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_cluster)

# ------------------- Plots -------------------

# Default vs non-default loans, visualizing class imbalance
plt.figure(figsize=(6, 4))
sns.countplot(x=y_default)
plt.title("Default vs Non-Default")
plt.grid(True)
plt.show()

# ROC curve - to see how well my classifier separates the defaulters from the rest
fpr, tpr, _ = roc_curve(y_test_cls, prob_cls)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test_cls, prob_cls):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted Default Amount - sanity check for regression
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test_reg, y=pred_reg_lgb, alpha=0.4)
plt.title("Actual vs Predicted Default Amount")
plt.grid(True)
plt.show()

# Exploring distribution of Expected Loss — where do we stand to lose the most?
plt.figure(figsize=(6, 4))
sns.histplot(df_result["Expected_Loss"], bins=50, kde=True)
plt.title("Expected Loss Distribution")
plt.grid(True)
plt.show()

# Clustering insights: Are certain income/FICO groups riskier than others?

# KMeans clustering results
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['fico_score'], y=df['log_annual_inc'], hue=df['KMeans_Cluster'], palette='Set2')
plt.title("KMeans Clustering (FICO vs Income)")
plt.grid(True)
plt.show()

# DBSCAN clustering results
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['fico_score'], y=df['log_annual_inc'], hue=df['DBSCAN_Cluster'], palette='tab10')
plt.title("DBSCAN Clustering (FICO vs Income)")
plt.grid(True)
plt.show()

# ------------------- Cleanup -------------------

#Freeing up memory before wrapping up
del X_train_scaled, X_test_scaled, y_train_cls, y_test_cls, y_train_reg, y_test_reg
gc.collect()

print("\nAll models trained, evaluated, and visualized successfully.")
