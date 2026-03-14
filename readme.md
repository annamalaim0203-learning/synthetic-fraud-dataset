# 💳 Synthetic Fraud Detection – Machine Learning Financial Domain Project

---

# 📌 Project Overview

This project demonstrates the **development and implementation of multiple machine learning algorithms to detect fraudulent banking transactions** using transaction data.

Fraud detection is one of the most critical applications of machine learning in the **financial and banking domain**. Financial institutions process millions of transactions daily, making manual monitoring impossible.

Machine learning models analyze patterns in transaction data and automatically identify **suspicious or fraudulent activities**.

This project implements a **complete end-to-end machine learning workflow** including:

• Dataset loading
• Data preprocessing and feature engineering
• Implementation of multiple machine learning models
• Model evaluation using several metrics
• Explainable AI using SHAP
• Deployment through a Streamlit interactive web application

---

# 🎯 Project Objectives

The project addresses the following goals:

**1. Develop and implement machine learning algorithms for financial domain applications**

Multiple machine learning models were implemented to detect fraudulent transactions.

**2. Utilize bias and variance techniques to enhance model performance**

Different models with varying complexity were used to balance bias and variance to improve prediction reliability.

---

# 📊 Dataset Description

Dataset Name: Synthetic Fraud Detection Dataset
File Used: synthetic_fraud_dataset.csv
Problem Type: Binary Classification

Target Variable: **is_fraud**

Target Meaning

1 → Fraudulent Transaction
0 → Legitimate Transaction

Number of Records: ~10,000
Number of Features: 9 features + 1 target variable

---

# Dataset Features

transaction_id – Unique transaction identifier
user_id – Unique user identifier
amount – Transaction amount
transaction_type – Type of transaction (ATM / POS / QR / Online)
merchant_category – Category of merchant
country – Country where transaction occurred
hour – Hour of transaction
device_risk_score – Device fraud risk score
ip_risk_score – IP fraud risk score
is_fraud – Fraud label

---

# ⚙️ Machine Learning Models Used

Six machine learning models were implemented and compared.

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest
6. XGBoost

Each model represents a **different bias-variance tradeoff**.

---

# 📊 Evaluation Metrics Used

To evaluate fraud detection performance, multiple metrics were used.

Accuracy – Overall prediction correctness
Precision – How many predicted frauds are actually fraud
Recall – How many real frauds were detected
F1 Score – Balance between precision and recall
AUC Score – Model ability to distinguish fraud vs legitimate
MCC – Balanced metric for imbalanced datasets

These metrics help provide **a more reliable evaluation of model performance**.

---

# ⚖️ Bias and Variance Explanation

## What is Bias?

Bias occurs when a model is **too simple** and cannot learn complex patterns in the data.

Example:
A fraud model that only checks transaction amount.

Result: **Underfitting**

---

## What is Variance?

Variance occurs when a model **memorizes training data** instead of learning general patterns.

Example:
A model that remembers specific transactions instead of fraud behavior.

Result: **Overfitting**

---

## Bias-Variance Usage in This Project

Different models were selected to balance bias and variance.

Logistic Regression → High Bias, Low Variance
Decision Tree → Low Bias, High Variance
Random Forest → Balanced Bias and Variance
XGBoost → Low Bias with controlled variance

Ensemble models such as **Random Forest and XGBoost help reduce variance while maintaining strong predictive power**.

---

# 🧠 Explainable AI – SHAP

Machine learning models often behave like **black boxes**.

SHAP (SHapley Additive Explanations) helps explain:

• Which features influenced the prediction
• How strongly each feature affected the result
• Why the model predicted fraud

This is important for **financial systems where transparency is required**.

---

# 🧩 Simple Breakdown of app.py

The entire fraud detection system is implemented inside **app.py**.

Below is a simple explanation of each part of the code.

---

## 1. Importing Libraries

The application begins by importing required libraries.

These libraries provide functionality for:

• Web interface (Streamlit)
• Data manipulation (Pandas, NumPy)
• Visualization (Matplotlib, Seaborn)
• Machine learning models (Scikit-Learn, XGBoost)
• Explainable AI (SHAP)

---

## 2. Streamlit Page Configuration

The Streamlit dashboard is configured using:

st.set_page_config()

This sets:

• Page title
• Layout of the dashboard

The application title is then displayed using:

st.title()

---

## 3. Dataset Download from GitHub

The application automatically loads the dataset from a GitHub repository using:

requests.get()

Users can also download the dataset directly through the dashboard.

---

## 4. Sidebar Controls

The Streamlit sidebar allows users to:

Upload a dataset
Select a machine learning model

This makes the application interactive.

---

## 5. Dataset Overview Section

After loading the dataset, the application displays:

• Number of rows
• Number of columns
• Target variable
• Sample records

This helps users understand the dataset structure.

---

## 6. Model Explanation Section

When a user selects a model from the dropdown, the application displays:

• Definition of the model
• Use case
• Simple analogy
• How the model helps detect fraud

This section helps users understand machine learning models easily.

---

## 7. Data Preprocessing

Before training models, the dataset is cleaned.

Steps include:

Removing identifier columns
Encoding categorical features
Preparing features for machine learning algorithms

This ensures models receive **clean and structured data**.

---

## 8. Feature Engineering

New features are created to help the model detect fraud patterns.

Examples include:

High amount transactions
Night time transactions
International transactions

Feature engineering improves model learning.

---

## 9. Train-Test Split

The dataset is divided into:

Training Data – used to train the model
Testing Data – used to evaluate the model

This ensures the model performs well on **unseen transactions**.

---

## 10. Feature Scaling

Some algorithms require normalized data.

StandardScaler is used to scale numerical features so models perform better.

---

## 11. Model Selection

The selected machine learning model is loaded based on the dropdown choice.

Examples:

Logistic Regression
Decision Tree
Random Forest
XGBoost

The model is then trained using the training dataset.

---

## 12. Model Training

The selected model learns patterns from transaction data.

The trained model is then used to predict fraud on the test dataset.

---

## 13. Model Evaluation

After predictions are made, performance metrics are calculated.

These metrics include:

Accuracy
Precision
Recall
F1 Score
AUC Score
MCC

These metrics measure how well the model detects fraud.

---

## 14. Confusion Matrix

A confusion matrix visualizes prediction results.

It shows:

True Positives
True Negatives
False Positives
False Negatives

This helps understand model mistakes.

---

## 15. SHAP Explainability

Finally, SHAP is used to explain model predictions.

The SHAP plot shows:

• Feature importance
• Which variables influenced fraud prediction

This improves **model transparency**.

---

# 🌐 Streamlit Application Features

The Streamlit dashboard includes:

Dataset upload option
Model selection dropdown
Real-time model training
Performance metrics visualization
Confusion matrix visualization
SHAP explainability graphs

Users can interact with the system without writing any code.

---

# 🗂️ Project Structure

project-folder

app.py
requirements.txt
README.md

model/

logistic_model.py
decision_tree.py
knn.py
naive_bayes.py
random_forest.py
xgboost.py

---

# ⚙️ How to Run the Application

Step 1

Install dependencies

pip install -r requirements.txt

Step 2

Run Streamlit

streamlit run app.py

The dashboard will open automatically in the browser.

---

# 🚧 Challenges Faced

Several challenges occurred during development.

Dataset imbalance
Small test datasets
Model overfitting
SHAP compatibility issues with XGBoost

---

# ⚠️ Limitations

Synthetic dataset may not reflect real banking data
Fraud patterns vary across institutions
Small datasets may affect evaluation metrics

---

# 🚀 Future Improvements

Hyperparameter tuning
Cross validation
Real-time fraud detection pipeline
Large scale financial datasets
Advanced explainability techniques

---

# 🏦 Real-World Applications

Fraud detection models are widely used in:

Credit card fraud detection
Online banking monitoring
Payment gateway security
Anti-money laundering systems

Machine learning helps banks **detect suspicious activity quickly and automatically**.

---

# Explainability, Privacy, and Fairness in the Model

## 1. Explainability

**What it means**

Explainability refers to the ability to understand **why a machine learning model made a particular decision**.

**How it is used in this project**

In this project, Explainable AI is implemented using **SHAP (SHapley Additive Explanations)**. SHAP helps identify which features influenced the fraud prediction and how strongly they contributed to the model’s decision.

This allows users to understand:

- Which transaction attributes influenced fraud detection  
- How each feature contributed to the prediction  
- Why the model classified a transaction as fraudulent  

This improves **transparency and trust**, which is especially important in financial systems.

**Simple Analogy**

Imagine a **bank officer rejecting a loan application**.  
Instead of simply saying *"Loan rejected"*, the officer explains:

- Credit score is low  
- Income is insufficient  
- Existing debts are high  

SHAP performs a similar role by explaining **why the model flagged a transaction as fraud**.

---

## 2. Privacy

**What it means**

Privacy refers to protecting **sensitive personal information** while using data for machine learning.

**How it is used in this project**

To maintain privacy, the model does not rely on personal identifiers. The following columns were removed before training:

- transaction_id  
- user_id  

The model instead focuses on **behavioral patterns**, such as:

- transaction amount  
- transaction type  
- country  
- time of transaction  

This ensures the model detects fraud **without exposing personal identity**.

**Simple Analogy**

Imagine a **bank analyst reviewing suspicious transactions**.

Instead of seeing the customer's name, the analyst only sees:

- transaction amount  
- location  
- time of transaction  

The analyst can still detect suspicious activity **without knowing who the person is**, protecting customer privacy.

---

## 3. Fairness

**What it means**

Fairness ensures that machine learning models **do not make biased or discriminatory decisions**.

**How it is used in this project**

The fraud detection model uses **transaction behavior features** rather than personal attributes. The model evaluates patterns such as:

- transaction amount  
- merchant category  
- device risk score  
- transaction timing  

Sensitive attributes like **gender, race, or religion** are not included in the dataset. This reduces the risk of biased predictions.

**Simple Analogy**

Imagine **airport security screening passengers**.

A fair system checks:

- suspicious behavior  
- unusual travel patterns  
- baggage activity  

Instead of judging someone based on **appearance or nationality**.

Similarly, this fraud detection model focuses only on **transaction behavior**, ensuring fair predictions.

# 📌 Conclusion

This project demonstrates how machine learning algorithms can be applied to fraud detection in the financial domain.

By combining multiple models, feature engineering, bias-variance optimization, and explainable AI, the system provides a **comprehensive fraud detection solution**.

---

# 👨‍💻 Author

Machine Learning Financial Domain Project
Fraud Detection using Machine Learning and Explainable AI
