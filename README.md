# 📊 Telecom Customer Churn Analysis & Prediction System

An end-to-end Data Science and Machine Learning project that analyzes customer attrition patterns, identifies key flight risk drivers, and predicts future churn using machine learning with an interactive executive dashboard.

---

## 📌 1. Project Overview

Customer churn is a critical metric for telecom service providers. This project uses a dataset of **1,000 customer records** to:
1. Uncover behavioral and contractual factors that cause customers to leave.
2. Train and validate a **Random Forest Classifier** achieving **99.5% accuracy**.
3. Provide a real-time **Customer ID lookup & inference tool**.
4. Deliver an **Interactive Executive Web Dashboard** with real-time risk simulation.

---

## 🔍 2. Key Business Insights

- **Overall Churn Rate**: **88.3%** of customers in the dataset churned (883 churned vs 117 retained).
- **Contract Type Impact**: **100% of Month-to-Month customers churned**. Long-term contracts (1-Year and 2-Year) had significantly higher customer retention.
- **Tech Support Necessity**: **100% of customers without Tech Support churned**. All 117 retained customers had active Tech Support.
- **Tenure & Billing Danger Zone**: New customers (**Tenure < 12 months**) paying high monthly bills (**> $70/month**) represent the highest flight risk.
- **Demographics**: Gender has no statistically significant impact on churn (~87.4% female vs ~89.4% male).

---

## 📂 3. Project Structure

```text
TelecomChurn/
│
├── churn_data.csv            # Raw dataset (1,000 customer records)
├── churn_analysis.py         # Core pipeline: EDA, ML training, evaluation, graph generation
├── predict_churn.py          # CLI prediction tool: Look up any Customer ID or test custom profiles
├── dashboard.html            # Standalone interactive executive dashboard with lightbox viewer
├── README.md                 # Project documentation
│
└── outputs/
    ├── best_churn_model.pkl  # Serialized trained Random Forest model artifact
    └── plots/                # Curated essential analysis graphs
        ├── 1_contract_churn_rates.png   # 100% Stacked bar chart of churn by contract
        ├── 2_tenure_vs_charges.png      # Customer risk map (Tenure vs Monthly Charges)
        ├── 3_feature_importance.png     # Ranked top ML predictors (Horizontal bar)
        └── 4_confusion_matrix.png       # Model accuracy validation matrix
```

---

## ⚡ 4. Quick Start & How to Run

### Prerequisites
Make sure you have Python 3.9+ installed along with the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Step 1: Run the Complete Data Science Pipeline
Executes exploratory analysis, trains the Random Forest model, and generates the 4 essential plots:

```bash
python churn_analysis.py
```

### Step 2: Predict Churn for Any Customer ID
Look up real customer records from the dataset by their `CustomerID` (e.g. #1, #9, #26):

```bash
python predict_churn.py
```

*Example Output:*
```text
=======================================================
 CHURN PREDICTION FOR CUSTOMER ID #9
=======================================================
  • CustomerID      : 9
  • Age             : 40
  • Gender          : Female
  • Tenure          : 53 months
  • MonthlyCharges  : $49.81
  • ContractType    : Two-Year
  • InternetService : Fiber Optic
  • TechSupport     : Yes
-------------------------------------------------------
  • Actual Status   : No (Retained)
  • Model Prediction: Will Stay (No)
  • Churn Risk Score: 17.05%
=======================================================
```

### Step 3: Open the Interactive Executive Dashboard
Simply open `dashboard.html` in any web browser (Google Chrome, Edge, Firefox, Safari):
- **KPI Summary Cards**: High-level operational metrics.
- **Interactive Chart.js Charts**: Dynamic visual exploration.
- **Live Customer Risk Simulator**: Adjust parameters to calculate real-time churn risk.
- **Graph Lightbox Viewer**: Click any chart card to view in full resolution with `← Back to Dashboard` navigation.

---

## 📈 5. Machine Learning Model Performance

Evaluated on an **80/20 stratified train-test split** (200 unseen test samples):

| Metric | Score |
| :--- | :---: |
| **Model** | Random Forest Classifier (100 estimators, max_depth=6) |
| **Test Accuracy** | **99.50%** |
| **Precision (Churn)** | **99.44%** |
| **Recall (Churn)** | **100.00%** |
| **F1-Score** | **0.9972** |
| **ROC-AUC** | **1.0000** |

### Top Predictive Feature Ranking (Gini Importance)
1. `TechSupport` (20.0%)
2. `Tenure` (18.1%)
3. `ContractType: One-Year` (15.7%)
4. `MonthlyCharges` (15.7%)
5. `ContractType: Two-Year` (11.7%)

---

## 💡 6. Recommended Retention Strategies

1. **Phase out Month-to-Month lock-in barriers**: Offer a 10–15% onboarding discount to transition customers into 1-Year or 2-Year contracts.
2. **Bundle Complimentary Tech Support**: Since 100% of non-supported customers churned, include basic tech support in all standard tiers.
3. **Target Early Tenure (Months 1–6)**: Deploy automated health-check calls and proactive support during the first 90 days.
4. **Price Optimization**: For accounts paying > $80/month, proactively propose custom bundle adjustments to avoid bill fatigue.
