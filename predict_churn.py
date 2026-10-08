# ==============================================================================
# PREDICT CUSTOMER CHURN FOR ANY CUSTOMER (BY ID OR MANUAL INPUT)
# Level: Intermediate Python
# ==============================================================================

import joblib
import pandas as pd

# 1. Load the trained model package
model_package = joblib.load('outputs/best_churn_model.pkl')
model = model_package['model']
expected_features = model_package['feature_names']


def predict_churn(customer_dict):
    """
    Takes a single customer dictionary, formats the columns,
    and returns whether they will churn along with their churn probability.
    """
    # Create an empty DataFrame with all expected model features, initialized to 0
    df_input = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # 1. Fill in numerical features
    numeric_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        if col in customer_dict:
            df_input[col] = customer_dict[col]
            
    # 2. Set one-hot encoded categorical columns to 1 if present
    categorical_cols = ['Gender', 'ContractType', 'InternetService', 'TechSupport']
    for col in categorical_cols:
        val = customer_dict.get(col)
        encoded_column_name = f"{col}_{val}"
        if encoded_column_name in expected_features:
            df_input[encoded_column_name] = 1

    # Predict probability of churn (class 1)
    churn_probability = model.predict_proba(df_input)[0, 1]
    
    # Determine classification result
    churn_prediction = "Will Churn (Yes)" if churn_probability >= 0.50 else "Will Stay (No)"
    
    return churn_prediction, churn_probability


def predict_by_customer_id(customer_id, csv_file='churn_data.csv'):
    """
    Looks up a real customer from the CSV file by their CustomerID and predicts their churn risk.
    """
    df = pd.read_csv(csv_file, keep_default_na=False)
    customer_row = df[df['CustomerID'] == customer_id]
    
    if customer_row.empty:
        print(f"❌ Error: Customer ID {customer_id} not found in {csv_file}")
        return
    
    cust = customer_row.iloc[0].to_dict()
    actual_status = cust.pop('Churn', 'Unknown')
    
    pred, prob = predict_churn(cust)
    
    print("=" * 55)
    print(f" CHURN PREDICTION FOR CUSTOMER ID #{customer_id}")
    print("=" * 55)
    for key, value in cust.items():
        print(f"  • {key:<16}: {value}")
    print("-" * 55)
    print(f"  • Actual Status   : {actual_status}")
    print(f"  • Model Prediction: {pred}")
    print(f"  • Churn Risk Score: {prob * 100:.2f}%")
    print("=" * 55)


# ==============================================================================
# DEMONSTRATION & USAGE
# ==============================================================================
if __name__ == '__main__':
    print("==================================================")
    print("      TELECOM CUSTOMER CHURN PREDICTOR           ")
    print("==================================================")
    
    # Example 1: Look up a real customer from the dataset by CustomerID
    # Customer #1 (Churned Month-to-Month customer)
    predict_by_customer_id(customer_id=1)
    print()
    
    # Customer #9 (Retained Two-Year contract customer)
    predict_by_customer_id(customer_id=9)
    print()
    
    # Customer #26 (Retained Two-Year contract customer)
    predict_by_customer_id(customer_id=26)
