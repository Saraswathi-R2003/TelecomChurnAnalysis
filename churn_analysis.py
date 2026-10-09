# ==============================================================================
# TELECOM CUSTOMER CHURN ANALYSIS & PREDICTION
# Level: Intermediate Python (Concise & Focused)
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Create output folder for graphs
os.makedirs('outputs/plots', exist_ok=True)
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------------------------
print("--- 1. Loading Dataset ---")
df = pd.read_csv('churn_data.csv', keep_default_na=False)
print(f"Loaded {len(df)} customer records.")
print(f"Overall Churn Rate: {(df['Churn'] == 'Yes').mean() * 100:.1f}%\n")

# ------------------------------------------------------------------------------
# 2. GENERATE THE 4 ESSENTIAL & IMPORTANT GRAPHS
# ------------------------------------------------------------------------------
print("--- 2. Generating Essential Visualizations ---")

# Graph 1: Churn Rate by Contract Type (100% Stacked Bar)
plt.figure(figsize=(7, 4.5))
contract_churn = pd.crosstab(df['ContractType'], df['Churn'], normalize='index') * 100
contract_churn[['No', 'Yes']].plot(
    kind='bar', stacked=True, color=['#10b981', '#ef4444'], edgecolor='white', figsize=(7, 4.5)
)
plt.title('Churn Rate by Contract Type (%)', fontsize=12, fontweight='bold')
plt.ylabel('Percentage of Customers (%)')
plt.xlabel('Contract Type')
plt.legend(['Retained', 'Churned'], loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/plots/1_contract_churn_rates.png', dpi=300)
plt.close()
print("Saved: outputs/plots/1_contract_churn_rates.png")

# Graph 2: Customer Risk Map (Tenure vs Monthly Charges)
plt.figure(figsize=(7, 4.5))
sns.scatterplot(
    data=df, x='Tenure', y='MonthlyCharges', hue='Churn',
    palette={'Yes': '#ef4444', 'No': '#10b981'}, alpha=0.75
)
plt.title('Tenure vs Monthly Charges (Risk Distribution)', fontsize=12, fontweight='bold')
plt.xlabel('Tenure (Months)')
plt.ylabel('Monthly Charges ($)')
plt.tight_layout()
plt.savefig('outputs/plots/2_tenure_vs_charges.png', dpi=300)
plt.close()
print("Saved: outputs/plots/2_tenure_vs_charges.png")

# ------------------------------------------------------------------------------
# 3. DATA PREPROCESSING & MODEL TRAINING
# ------------------------------------------------------------------------------
print("\n--- 3. Training Machine Learning Model ---")

# Encode target and categorical columns
data = df.drop(columns=['CustomerID'])
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
data = pd.get_dummies(data, drop_first=True, dtype=int)

X = data.drop(columns=['Churn'])
y = data['Churn']
feature_names = X.columns.tolist()

# 80% Train, 20% Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Test Accuracy: {accuracy * 100:.2f}%\n")

# ------------------------------------------------------------------------------
# 4. MODEL EVALUATION GRAPHS
# ------------------------------------------------------------------------------
print("--- 4. Saving Model Evaluation Graphs ---")

# Graph 3: Top Feature Importance (Horizontal Bar)
importance_df = pd.DataFrame({
    'Feature': [f.replace('_', ': ') for f in feature_names],
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8, 4.5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='#2563eb')
plt.title('Top Factors Driving Customer Churn', fontsize=12, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/plots/3_feature_importance.png', dpi=300)
plt.close()
print("Saved: outputs/plots/3_feature_importance.png")

# Graph 4: Confusion Matrix Heatmap
plt.figure(figsize=(5.5, 4.5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Retain', 'Predicted Churn'],
            yticklabels=['Actual Retain', 'Actual Churn'])
plt.title(f'Confusion Matrix ({accuracy*100:.1f}% Accuracy)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/4_confusion_matrix.png', dpi=300)
plt.close()
print("Saved: outputs/plots/4_confusion_matrix.png")

# ------------------------------------------------------------------------------
# 5. SAVE TRAINED MODEL
# ------------------------------------------------------------------------------
joblib.dump({'model': rf_model, 'feature_names': feature_names}, 'outputs/best_churn_model.pkl')
print("\n[+] Model saved to 'outputs/best_churn_model.pkl'")
print("[+] All 4 essential graphs saved in 'outputs/plots/' successfully!")
