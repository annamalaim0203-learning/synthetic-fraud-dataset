import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Synthetic Fraud Detection Dashboard")
st.markdown("### Fraud Detection with Explainable AI (SHAP)")


# --------------------------------------------------
# DATASET DOWNLOAD
# --------------------------------------------------

st.subheader("🐙 Dataset Access – GitHub")

GITHUB_RAW_URL = "https://raw.githubusercontent.com/annamalaim0203-learning/synthetic-fraud-dataset/main/synthetic_fraud_dataset.csv"

response = requests.get(GITHUB_RAW_URL)

st.download_button(
    label="Download Fraud Dataset",
    data=response.content,
    file_name="synthetic_fraud_dataset.csv",
    mime="text/csv"
)

st.markdown("---")


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Fraud Dataset",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file is None:
    st.info("Upload dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader("Dataset Overview")

summary = pd.DataFrame({
    "Attribute": ["Rows", "Columns", "Target"],
    "Value": [df.shape[0], df.shape[1], "is_fraud"]
})

st.table(summary)

st.write("Sample Records")
st.dataframe(df.head())


# --------------------------------------------------
# MODEL EXPLANATION SECTION
# --------------------------------------------------

st.markdown("---")
st.subheader("📘 Model Explanation")

model_explanations = {

"Logistic Regression": {
"definition":
"Logistic Regression predicts the probability of a binary outcome such as fraud or non-fraud.",

"use_case":
"Used in credit risk prediction, fraud detection, and medical diagnosis.",

"analogy":
"Like a bank analyst estimating the probability that a transaction is suspicious based on several risk factors.",

"fraud_use":
"It helps estimate the probability that a transaction is fraudulent."
},

"Decision Tree": {
"definition":
"A Decision Tree splits data into branches based on conditions until a final classification is reached.",

"use_case":
"Used in loan approval systems and risk assessment.",

"analogy":
"Like a flowchart used by banks to approve or reject transactions.",

"fraud_use":
"It helps detect fraud by applying sequential rules on transaction attributes."
},

"KNN": {
"definition":
"K-Nearest Neighbors classifies a transaction based on similar past transactions.",

"use_case":
"Used in recommendation systems and anomaly detection.",

"analogy":
"Like asking nearby customers whether a transaction looks suspicious.",

"fraud_use":
"It detects fraud by comparing a transaction to similar past transactions."
},

"Naive Bayes": {
"definition":
"Naive Bayes calculates probabilities using Bayes theorem assuming feature independence.",

"use_case":
"Used in spam detection and document classification.",

"analogy":
"Like combining clues to estimate the likelihood of fraud.",

"fraud_use":
"It estimates fraud probability based on independent transaction features."
},

"Random Forest": {
"definition":
"Random Forest combines multiple decision trees and takes a majority vote.",

"use_case":
"Used in credit scoring and fraud detection.",

"analogy":
"Like asking multiple bank analysts to evaluate a transaction.",

"fraud_use":
"It improves fraud detection accuracy using ensemble learning."
},

"XGBoost": {
"definition":
"XGBoost is an advanced boosting algorithm where each tree improves the previous model.",

"use_case":
"Used in financial risk modeling and fraud detection competitions.",

"analogy":
"Like repeatedly correcting mistakes to improve fraud detection.",

"fraud_use":
"It identifies complex fraud patterns in transaction data."
}
}

info = model_explanations[model_name]

st.write("**Definition:**", info["definition"])
st.write("**Use Case:**", info["use_case"])
st.write("**Simple Analogy:**", info["analogy"])
st.write("**Application in Fraud Detection:**", info["fraud_use"])


# --------------------------------------------------
# REMOVE LEAKAGE FEATURES
# --------------------------------------------------

df = df.drop(columns=[
    "transaction_id",
    "user_id",
    "device_risk_score",
    "ip_risk_score"
], errors="ignore")


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

df["high_amount_flag"] = (df["amount"] > df["amount"].median()).astype(int)
df["night_transaction"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
df["international_transaction"] = (df["country"] != "US").astype(int)


# --------------------------------------------------
# FEATURES / TARGET
# --------------------------------------------------

target = "is_fraud"

X = df.drop(target, axis=1)
y = df[target]

X = pd.get_dummies(X, drop_first=True)


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

# Check if both classes exist
if len(y.unique()) > 1:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )


# --------------------------------------------------
# SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

models = {
"Logistic Regression": LogisticRegression(max_iter=1000),
"Decision Tree": DecisionTreeClassifier(),
"KNN": KNeighborsClassifier(),
"Naive Bayes": GaussianNB(),
"Random Forest": RandomForestClassifier(),
"XGBoost": XGBClassifier(eval_metric="logloss")
}

model = models[model_name]


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

if model_name in ["Logistic Regression","KNN","Naive Bayes"]:

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]

else:

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]


# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", round(accuracy,4))
c2.metric("Precision", round(precision,4))
c3.metric("F1 Score", round(f1,4))

c1.metric("AUC", round(auc,4))
c2.metric("Recall", round(recall,4))
c3.metric("MCC", round(mcc,4))


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

st.pyplot(fig)


# --------------------------------------------------
# SHAP EXPLAINABILITY
# --------------------------------------------------

st.subheader("")

if model_name in ["Random Forest", "Decision Tree", "XGBoost"]:

    try:

        # Fix specifically for XGBoost
        if model_name == "XGBoost":
            explainer = shap.TreeExplainer(model.get_booster())
        else:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_test)

        fig_shap = plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)

        st.pyplot(fig_shap)

    except Exception as e:
        st.warning("SHAP explanation could not be generated for this model.")

else:

    st.info("SHAP explanation is best supported for tree-based models.")



