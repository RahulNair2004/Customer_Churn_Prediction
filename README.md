# Customer Churn Prediction

![Churn Prediction](https://img.shields.io/badge/ML-Churn_Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning project to predict customer churn using historical customer data. This model helps businesses identify customers at risk of leaving, enabling proactive retention strategies.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation & Setup](#installation--setup)
- [Model Performance](#model-performance)
- [Insights](#insights)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

Customer churn (attrition) is a critical business metric. This project implements machine learning algorithms to predict which customers are likely to churn, allowing businesses to take targeted retention actions before losing customers.

The model analyzes customer demographics, service usage patterns, billing information, and engagement metrics to identify churn risk factors.

## ✨ Features

- **Data preprocessing pipeline** with handling for missing values and outliers
- **Feature engineering** to extract meaningful insights from raw customer data
- **Multiple ML models** comparison (Random Forest, XGBoost, Logistic Regression)
- **Hyperparameter tuning** for optimal model performance
- **Model interpretation** tools to understand churn factors
- **Visualization dashboard** for easy insight extraction
- **Prediction API** for integration with existing systems

## 📁 Directory Structure

```
customer-churn-prediction/
├── data/                     # Dataset files
│   ├── raw/                  # Original, immutable data
│              
├── notebooks/                # Jupyter notebooks for exploration and analysis
├── src/                      # Source code        
│   ├── models/               # Model training and evaluation
│   ├── visualization/        # EDA and results visualization
├── config/                   # Configuration files
├── requirements.txt          # Project dependencies
├── setup.py                  # Installation script
└── README.md                 # Project documentation
```

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Setup the configuration:
   ```bash
   cp config/config.example.yaml config/config.yaml
   # Edit config.yaml with your settings
   ```



Then send POST requests to `http://localhost:5000/predict` with customer data.

## 📊 Model Performance

Our best performing model achieves:
- **AUC-ROC**: 0.80
- **Precision**: 0.83
- **Recall**: 0.76
- **F1 Score**: 0.79


## 🧠 Insights

Key factors contributing to customer churn:
1. Contract duration (month-to-month contracts have higher churn)
2. Monthly charges (higher charges correlate with increased churn)
3. Customer service calls (more than 4 calls increases churn probability)
4. Tenure (customers with less than 6 months are at highest risk)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
