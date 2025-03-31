from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  

MODEL_DIR = r"C:\Users\Rahul\Desktop\DSA\Data_Science\Customer_Churn_Prediction\models"

# Loading the Models
log_reg_model_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
catboost_model_path = os.path.join(MODEL_DIR, "catboost_model.pkl")

log_reg_model = None
catboost_model = None

try:
    if os.path.exists(log_reg_model_path):
        log_reg_model = joblib.load(log_reg_model_path)
    else:
        print("Error: logistic_regression_model.pkl not found!")

    if os.path.exists(catboost_model_path):
        catboost_model = joblib.load(catboost_model_path)
    else:
        print("Error: catboost_model.pkl not found!")

    if log_reg_model and catboost_model:
        print("Models loaded successfully!")
    else:
        print("Warning: One or both models are missing!")

except Exception as e:
    print(f"Error loading models: {e}")

# Basic feature names from the dataset
BASE_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen',
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Feature engineering function
def engineer_features(df):
    # Create a copy of the og Dataframe
    df_processed = df.copy()
    
    # Convertinng SeniorCitizen column to numeric
    if df_processed['SeniorCitizen'].dtype == object:
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({'Yes': 1, 'No': 0})
    
    # Convert TotalCharges to numeric
    if df_processed['TotalCharges'].dtype == object:
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    
    # Average monthly charges
    df_processed['AvgMonthlyCharges'] = df_processed['MonthlyCharges']
    if df_processed['tenure'].max() > 0:  # Avoid division by zero
        df_processed['AvgMonthlyCharges'] = df_processed['TotalCharges'] / df_processed['tenure'].clip(lower=1)
    
    # Charges per tenure
    df_processed['ChargesPerTenure'] = df_processed['TotalCharges'] / df_processed['tenure'].clip(lower=1)
    
    # Customer value segments
    df_processed['IsHighValue'] = (df_processed['MonthlyCharges'] > 80).astype(int)
    
    # Tenure based columns
    df_processed['IsLongTermCustomer'] = (df_processed['tenure'] > 60).astype(int)
    df_processed['IsNewCustomer'] = (df_processed['tenure'] <= 12).astype(int)
    
    
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False)
    
    return df_encoded

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Customer Churn Prediction API", "status": "OK"})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not log_reg_model or not catboost_model:
        return jsonify({"error": "Models are not loaded"}), 500

    try:
        # Get JSON data from request
        data = request.json

       
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON format"}), 400

        # Convert to DataFrame with expected columns
        input_df = pd.DataFrame([data])

        # Checks all the Columns expecteed are in it
        for col in BASE_FEATURES:
            if col not in input_df:
                input_df[col] = np.nan  

        # Apply feature engineering
        processed_df = engineer_features(input_df)
        
        # Get the expected feature names from the model
        if hasattr(log_reg_model, 'feature_names_in_'):
            model_features = log_reg_model.feature_names_in_
        elif hasattr(catboost_model, 'feature_names_'):
            model_features = catboost_model.feature_names_
        else:
            
            print("Warning: Cannot determine feature names from models")
            model_features = processed_df.columns

        # To Check if all the columns are in the Dataframe
        for col in model_features:
            if col not in processed_df.columns:
                processed_df[col] = 0  

        
        processed_df = processed_df[model_features]

        # Make predictions
        log_reg_prob = log_reg_model.predict_proba(processed_df)[0][1]  
        catboost_prob = catboost_model.predict_proba(processed_df)[0][1]  

        # Average the predictions 
        ensemble_prob = (log_reg_prob + catboost_prob) / 2
        churn_prediction = 1 if ensemble_prob > 0.5 else 0

        # Response to be Generated
        response = {
            'churn_probability': float(ensemble_prob),
            'churn_prediction': int(churn_prediction),
            'model_details': {
                'logistic_regression_prob': float(log_reg_prob),
                'catboost_prob': float(catboost_prob)
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': log_reg_model is not None and catboost_model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)