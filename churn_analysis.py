# ==============================================================================
# TELECOM CUSTOMER CHURN ANALYSIS & PREDICTION
# Level: Intermediate Python / Data Science
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI popup issues when saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Create a clean folder to save our essential graphs
os.makedirs('outputs/plots', exist_ok=True)

# Set clean visualization style
sns.set_theme(style="whitegrid", font="DejaVu Sans")
palette_colors = {'Yes': '#e74c3c', 'No': '#2ecc71'}

# ==============================================================================
# STEP 1: LOAD AND INSPECT THE DATASET
# ==============================================================================
print("--- STEP 1: Loading Dataset ---")
df = pd.read_csv('churn_data.csv', keep_default_na=False)

print(f"Total Customers: {df.shape[0]}")
print(f"Churn Rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%\n")


# ==============================================================================
# STEP 2: GENERATE ESSENTIAL, EASY-TO-UNDERSTAND GRAPHS
# ==============================================================================
print("--- STEP 2: Creating Essential Visualizations ---")

# --- GRAPH 1: Top 2 Categorical Drivers (Contract Type & Tech Support) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1. Contract Type
sns.countplot(ax=axes[0], data=df, x='ContractType', hue='Churn', palette=palette_colors)
axes[0].set_title('Churn by Contract Type', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Contract Type')
axes[0].set_ylabel('Number of Customers')

# 2. Tech Support
sns.countplot(ax=axes[1], data=df, x='TechSupport', hue='Churn', palette=palette_colors)
axes[1].set_title('Churn by Tech Support Status', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Has Tech Support?')
axes[1].set_ylabel('Number of Customers')

plt.suptitle('Key Business Drivers of Customer Churn', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/plots/1_key_drivers.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/1_key_drivers.png")


# --- GRAPH 2: Tenure & Monthly Charges Impact ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1. Tenure Distribution
sns.histplot(ax=axes[0], data=df, x='Tenure', hue='Churn', kde=True, palette=palette_colors, bins=25)
axes[0].set_title('Customer Tenure (Months) vs Churn', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Tenure (Months with Company)')
axes[0].set_ylabel('Customer Count')

# 2. Monthly Charges Distribution
sns.histplot(ax=axes[1], data=df, x='MonthlyCharges', hue='Churn', kde=True, palette=palette_colors, bins=25)
axes[1].set_title('Monthly Charges ($) vs Churn', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Monthly Charges ($)')
axes[1].set_ylabel('Customer Count')

plt.suptitle('Tenure & Pricing Dynamics', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/plots/2_tenure_and_pricing.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/2_tenure_and_pricing.png")


# ==============================================================================
# STEP 3: PREPARE DATA FOR MACHINE LEARNING
# ==============================================================================
print("\n--- STEP 3: Preprocessing Data ---")

data = df.drop(columns=['CustomerID'])
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = ['Gender', 'ContractType', 'InternetService', 'TechSupport']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)

X = data.drop(columns=['Churn'])
y = data['Churn']
feature_names = X.columns.tolist()

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ==============================================================================
# STEP 4: TRAIN MACHINE LEARNING MODEL
# ==============================================================================
print("--- STEP 4: Training Random Forest Classifier ---")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%\n")


# ==============================================================================
# STEP 5: FEATURE IMPORTANCE & CONFUSION MATRIX GRAPHS
# ==============================================================================
print("--- STEP 5: Saving Model Visualizations ---")

# --- GRAPH 3: Top Feature Importance Ranking ---
importance_df = pd.DataFrame({
    'Feature': [f.replace('_', ': ') for f in feature_names],
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(9, 5))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3498db', edgecolor='#2980b9')
plt.title('Top Factors Driving Customer Churn (Random Forest)', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
for bar in bars:
    plt.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.3f}', va='center', fontsize=9)
plt.xlim(0, max(importance_df['Importance']) * 1.15)
plt.tight_layout()
plt.savefig('outputs/plots/3_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/3_feature_importance.png")


# --- GRAPH 4: Model Accuracy & Confusion Matrix ---
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Retain (0)', 'Predicted Churn (1)'],
            yticklabels=['Actual Retain (0)', 'Actual Churned (1)'])
plt.title(f'Confusion Matrix (Test Accuracy: {accuracy*100:.1f}%)', fontsize=13, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('outputs/plots/4_model_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/4_model_accuracy.png")


# ==============================================================================
# STEP 6: SAVE TRAINED MODEL
# ==============================================================================
joblib.dump({'model': rf_model, 'feature_names': feature_names}, 'outputs/best_churn_model.pkl')
print("\n[+] Model saved to 'outputs/best_churn_model.pkl'")
print("[+] All 4 essential graphs saved in 'outputs/plots/' successfully!")
