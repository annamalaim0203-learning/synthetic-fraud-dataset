import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import requests

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

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
# LOAD TRAINING DATASET FROM GITHUB
# --------------------------------------------------

GITHUB_DATASET = "https://raw.githubusercontent.com/annamalaim0203-learning/synthetic-fraud-dataset/main/synthetic_fraud_dataset.csv"

train_df = pd.read_csv(GITHUB_DATASET)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset",
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
    st.info("Upload sample dataset to test the model.")
    st.stop()

test_df = pd.read_csv(uploaded_file)


# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("Test Dataset Overview")

st.dataframe(test_df.head())


# --------------------------------------------------
# REMOVE LEAKAGE FEATURES
# --------------------------------------------------

train_df = train_df.drop(columns=[
    "transaction_id",
    "user_id",
    "device_risk_score",
    "ip_risk_score"
], errors="ignore")

test_df = test_df.drop(columns=[
    "transaction_id",
    "user_id",
    "device_risk_score",
    "ip_risk_score"
], errors="ignore")


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

for df in [train_df, test_df]:

    df["high_amount_flag"] = (df["amount"] > df["amount"].median()).astype(int)

    df["night_transaction"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

    df["international_transaction"] = (df["country"] != "US").astype(int)


# --------------------------------------------------
# FEATURES / TARGET
# --------------------------------------------------

target = "is_fraud"

X_train = train_df.drop(target, axis=1)
y_train = train_df[target]

X_test = test_df.drop(target, axis=1)
y_test = test_df[target]

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# --------------------------------------------------
# SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# MODELS
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

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

f1 = f1_score(y_test, y_pred, zero_division=0)

auc = roc_auc_score(y_test, y_prob)

mcc = matthews_corrcoef(y_test, y_pred)


# --------------------------------------------------
# DISPLAY METRICS
# --------------------------------------------------

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

st.subheader("Explainable AI – SHAP Feature Impact")

try:

    if model_name == "XGBoost":
        explainer = shap.TreeExplainer(model.get_booster())

    elif model_name in ["Random Forest","Decision Tree"]:
        explainer = shap.TreeExplainer(model)

    else:
        st.info("SHAP explanation available for tree models.")
        st.stop()

    shap_values = explainer.shap_values(X_test)

    fig = plt.figure()

    shap.summary_plot(shap_values, X_test, show=False)

    st.pyplot(fig)

except:

    st.warning("SHAP explanation could not be generated.")
