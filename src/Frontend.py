import streamlit as st
import requests
import pandas as pd
import json
import numpy as np

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Constants
API_URL = "http://localhost:5000/api/predict"
HEALTH_CHECK_URL = "http://localhost:5000/api/health"

# Check if the backend is running
def check_backend_health():
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "models_loaded": False}
    except:
        return {"status": "error", "models_loaded": False}

# Function to calculate derived features
def calculate_derived_features(data):
    # Copy the data to avoid modifying the original
    enhanced_data = data.copy()
    
    # Calculate AvgMonthlyCharges
    if data["tenure"] > 0:
        enhanced_data["AvgMonthlyCharges"] = data["TotalCharges"] / data["tenure"]
    else:
        enhanced_data["AvgMonthlyCharges"] = data["MonthlyCharges"]
    
    # Calculate ChargesPerTenure
    enhanced_data["ChargesPerTenure"] = data["MonthlyCharges"] / max(1, data["tenure"])
    
    # Calculate MonthlyToAvgRatio
    if enhanced_data["AvgMonthlyCharges"] > 0:
        enhanced_data["MonthlyToAvgRatio"] = data["MonthlyCharges"] / enhanced_data["AvgMonthlyCharges"]
    else:
        enhanced_data["MonthlyToAvgRatio"] = 1.0
    
    # Calculate TotalServices
    services = [
        data["PhoneService"] == "Yes",
        data["MultipleLines"] == "Yes",
        data["InternetService"] != "No",
        data["OnlineSecurity"] == "Yes",
        data["OnlineBackup"] == "Yes",
        data["DeviceProtection"] == "Yes",
        data["TechSupport"] == "Yes",
        data["StreamingTV"] == "Yes",
        data["StreamingMovies"] == "Yes"
    ]
    enhanced_data["TotalServices"] = sum(services)
    
    # Calculate customer segments
    enhanced_data["IsNewCustomer"] = 1 if data["tenure"] <= 12 else 0
    enhanced_data["IsLongTermCustomer"] = 1 if data["tenure"] >= 36 else 0
    enhanced_data["IsHighValue"] = 1 if data["MonthlyCharges"] >= 80 else 0
    
    return enhanced_data

# Function to make prediction
def predict_churn(customer_data):
    try:
        # Calculate derived features
        enhanced_data = calculate_derived_features(customer_data)
        
        # Make the API request
        response = requests.post(API_URL, json=enhanced_data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Main app
def main():
    # Page title and description
    st.title("Customer Churn Prediction")
    st.write("Enter customer information to predict churn probability")
    
    # Check backend health
    health_status = check_backend_health()
    if health_status["status"] == "healthy" and health_status["models_loaded"]:
        st.success("Backend is connected and models are loaded")
    else:
        st.error("Backend is not available or models are not loaded properly")
        st.stop()
    
    # Create form for user input
    with st.form("prediction_form"):
        # Layout with columns for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
            st.subheader("Account Information")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure, step=100.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])

        with col2:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            
            # Only show multiple lines option if they have phone service
            multiple_lines = "No" if phone_service == "No" else st.selectbox(
                "Multiple Lines", ["No", "Yes"]
            )
            
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
            # Only show internet-dependent services if they have internet
            if internet_service != "No":
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            else:
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
        
        # Submit button
        submit_button = st.form_submit_button("Predict Churn")
    
    # Process form submission
    if submit_button:
        # Map UI selections to model features
        senior_citizen_int = 1 if senior_citizen == "Yes" else 0
        partner_binary = "Yes" if partner == "Yes" else "No"
        dependents_binary = "Yes" if dependents == "Yes" else "No"
        
        # Prepare data in the format expected by the model
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen_int,
            "Partner": partner_binary,
            "Dependents": dependents_binary,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        # Make prediction
        with st.spinner("Calculating churn probability..."):
            result = predict_churn(customer_data)
        
        # Display results
        if "error" in result:
            st.error(f"Error making prediction: {result['error']}")
            
            # Show diagnostic information to help debug
            with st.expander("Show diagnostic information"):
                enhanced_data = calculate_derived_features(customer_data)
                st.write("Basic features:")
                st.write(customer_data)
                st.write("Derived features:")
                derived_only = {k:v for k,v in enhanced_data.items() if k not in customer_data}
                st.write(derived_only)
        else:
            churn_prob = result.get('churn_probability', 0) * 100
            churn_pred = result.get('churn_prediction', 0)
            
            # Display prediction result
            st.subheader("Prediction Results")
            
            # Columns for metrics and details
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                # Main prediction
                st.metric(
                    label="Churn Probability", 
                    value=f"{churn_prob:.1f}%",
                    delta=f"{'-' if churn_pred == 0 else '+'}{abs(churn_prob - 50):.1f}% from threshold"
                )
                
                # Prediction interpretation
                if churn_pred == 1:
                    st.error("❌ This customer is likely to churn")
                else:
                    st.success("✅ This customer is likely to stay")
            
            with results_col2:
                # Model details
                st.subheader("Model Details")
                model_details = result.get('model_details', {})
                log_reg_prob = model_details.get('logistic_regression_prob', 0) * 100
                catboost_prob = model_details.get('catboost_prob', 0) * 100
                
                st.write(f"Logistic Regression: {log_reg_prob:.1f}%")
                st.write(f"CatBoost: {catboost_prob:.1f}%")
                st.write(f"Ensemble (Average): {churn_prob:.1f}%")
            
            # Get the derived features
            enhanced_data = calculate_derived_features(customer_data)
            
            # Feature importance visualization
            st.subheader("Customer Profile Analysis")
            
            # Show some key risk factors
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Basic Metrics:")
                metrics = {
                    "Tenure": f"{tenure} months",
                    "Monthly Charges": f"${monthly_charges:.2f}",
                    "Total Charges": f"${total_charges:.2f}",
                    "Total Services": f"{enhanced_data['TotalServices']} services",
                    "Contract Type": contract
                }
                for key, value in metrics.items():
                    st.write(f"- {key}: {value}")
            
            with col2:
                st.write("Derived Metrics:")
                derived_metrics = {
                    "Avg Monthly Charges": f"${enhanced_data['AvgMonthlyCharges']:.2f}",
                    "Monthly to Avg Ratio": f"{enhanced_data['MonthlyToAvgRatio']:.2f}",
                    "Charges Per Tenure": f"${enhanced_data['ChargesPerTenure']:.2f}",
                    "Customer Segment": "New" if enhanced_data['IsNewCustomer'] else 
                                      ("Long-term" if enhanced_data['IsLongTermCustomer'] else "Mid-term"),
                    "Value Tier": "High value" if enhanced_data['IsHighValue'] else "Standard value"
                }
                for key, value in derived_metrics.items():
                    st.write(f"- {key}: {value}")
            
            # Risk factors
            st.subheader("Risk Factors")
            risk_factors = {
                "Month-to-Month Contract": 1 if contract == "Month-to-month" else 0,
                "High Monthly Charges": 1 if monthly_charges > 70 else 0,
                "No Online Security": 1 if online_security == "No" else 0,
                "No Tech Support": 1 if tech_support == "No" else 0,
                "Fiber Optic Internet": 1 if internet_service == "Fiber optic" else 0,
                "Electronic Check Payment": 1 if payment_method == "Electronic check" else 0,
                "New Customer (< 12 mo)": 1 if enhanced_data["IsNewCustomer"] == 1 else 0,
                "Paperless Billing": 1 if paperless_billing == "Yes" else 0,
                "High Monthly to Avg Ratio": 1 if enhanced_data["MonthlyToAvgRatio"] > 1.1 else 0
            }
            
            # Converting risk factor to DataFrame 
            risk_df = pd.DataFrame({
                'Risk Factor': list(risk_factors.keys()),
                'Present': list(risk_factors.values())
            })
            
            # Only show present risk factors
            present_risks = risk_df[risk_df['Present'] == 1]
            
            if len(present_risks) > 0:
                st.write("Risk factors present:")
                for factor in present_risks['Risk Factor']:
                    st.write(f"- {factor}")
            else:
                st.write("No major risk factors detected.")
            
            # Customer recommendations based on prediction
            st.subheader("Recommended Actions")
            if churn_pred == 1:
                st.write("Consider the following retention strategies:")
                if contract == "Month-to-month":
                    st.write("- Offer contract upgrade incentives")
                if internet_service == "Fiber optic" and (online_security == "No" or tech_support == "No"):
                    st.write("- Bundle internet with security and support services")
                if payment_method == "Electronic check":
                    st.write("- Encourage automatic payment methods")
                if enhanced_data["IsNewCustomer"] == 1:
                    st.write("- Provide special offers for new customers")
                if enhanced_data["IsHighValue"] == 1:
                    st.write("- High-touch customer service approach for this valuable customer")
                st.write("- Personalized outreach to understand pain points")
            else:
                st.write("This customer has a low churn risk. Consider the following:")
                st.write("- Opportunity for upselling additional services")
                st.write("- Loyalty rewards program enrollment")
                st.write("- Referral program incentives")
            
           
            with st.expander("View all customer data"):
                st.write("Basic features:")
                st.write(customer_data)
                st.write("Derived features:")
                derived_only = {k:v for k,v in enhanced_data.items() if k not in customer_data}
                st.write(derived_only)

# Run the app
if __name__ == "__main__":
    main()