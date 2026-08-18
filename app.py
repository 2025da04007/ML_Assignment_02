import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide"
)

st.title("Machine Learning Assignment 2")
st.subheader("Credit Card Default Prediction using Classification Models")

st.write("""
This Streamlit application evaluates multiple machine learning classification models
on uploaded credit card test data. The app displays evaluation metrics, confusion matrix,
classification report, and model comparison results.
""")

# ------------------------------------------------------------
# Model Folder and Model Files
# ------------------------------------------------------------

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

scaled_models = [
    "Logistic Regression",
    "KNN",
    "Naive Bayes"
]

target_column = "default.payment.next.month"

# ------------------------------------------------------------
# Load Saved Models
# ------------------------------------------------------------

@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_files[model_name])
    return joblib.load(model_path)


@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    return joblib.load(scaler_path)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("Project Sections")

section = st.sidebar.radio(
    "Select Section",
    [
        "Upload Test Data",
        "Model Evaluation",
        "Compare All Models"
    ]
)

# ------------------------------------------------------------
# Helper Function: Prepare Uploaded Data
# ------------------------------------------------------------

def prepare_uploaded_data(uploaded_df):
    """
    Prepares uploaded UCI Credit Card test data for model prediction.
    The uploaded file must contain default.payment.next.month.
    """

    df = uploaded_df.copy()

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Check target column
    if target_column not in df.columns:
        st.error(
            "Uploaded CSV must contain the target column: default.payment.next.month"
        )
        st.stop()

    # Drop ID column if available
    if "ID" in df.columns:
        df = df.drop("ID", axis=1)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    # Target conversion
    df[target_column] = df[target_column].astype(int)

    # Split features and target
    X_test = df.drop(target_column, axis=1)
    y_test = df[target_column]

    return X_test, y_test


# ------------------------------------------------------------
# Helper Function: Evaluate Model
# ------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):

    X_eval = X_test.copy()

    # Apply scaler only for models trained with scaled data
    if model_name in scaled_models:
        try:
            scaler = load_scaler()
            X_eval = scaler.transform(X_eval)
        except Exception as e:
            st.warning("Scaler not found or could not be applied. Continuing without scaling.")
            st.write(e)

    # Prediction
    y_pred = model.predict(X_eval)

    # Convert predictions to numeric 0/1
    y_pred = pd.Series(y_pred)
    y_pred = pd.to_numeric(y_pred, errors="coerce").fillna(0).astype(int)

    # Probability for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_test, y_pred)
    }

    return metrics, y_pred


# ------------------------------------------------------------
# Section 1: Upload Test Data
# ------------------------------------------------------------

if section == "Upload Test Data":

    st.header("Upload Test Data")

    st.write("""
    Upload the credit card `test_data.csv` file generated from the model notebook.

    The file must contain:
    - Feature columns such as LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, BILL_AMT1, PAY_AMT1, etc.
    - Target column: `default.payment.next.month`
    """)

    uploaded_file = st.file_uploader(
        "Upload test_data.csv",
        type=["csv"]
    )

    if uploaded_file is not None:

        test_df = pd.read_csv(uploaded_file)

        st.success("Test data uploaded successfully.")

        st.write("Dataset Shape:", test_df.shape)

        st.subheader("Preview of Uploaded Test Data")
        st.dataframe(test_df.head())

        st.subheader("Column Names")
        st.write(test_df.columns.tolist())

        st.subheader("Target Distribution")

        if target_column in test_df.columns:
            st.write(test_df[target_column].value_counts())
        else:
            st.error("Target column default.payment.next.month not found.")

    else:
        st.info("Please upload test_data.csv to continue.")


# ------------------------------------------------------------
# Section 2: Model Evaluation
# ------------------------------------------------------------

elif section == "Model Evaluation":

    st.header("Model Evaluation")

    uploaded_file = st.file_uploader(
        "Upload test_data.csv",
        type=["csv"]
    )

    if uploaded_file is not None:

        test_df = pd.read_csv(uploaded_file)

        X_test, y_test = prepare_uploaded_data(test_df)

        selected_model = st.selectbox(
            "Select Machine Learning Model",
            list(model_files.keys())
        )

        try:
            model = load_model(selected_model)

            metrics, y_pred = evaluate_model(
                model,
                X_test,
                y_test,
                selected_model
            )

            st.subheader("Evaluation Metrics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", round(metrics["Accuracy"], 4))
            col2.metric("AUC Score", round(metrics["AUC Score"], 4))
            col3.metric("Precision", round(metrics["Precision"], 4))

            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", round(metrics["Recall"], 4))
            col5.metric("F1 Score", round(metrics["F1 Score"], 4))
            col6.metric("MCC Score", round(metrics["MCC Score"], 4))

            # Confusion Matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm, cmap="Blues")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=14
                    )

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["No Default", "Default"])
            ax.set_yticklabels(["No Default", "Default"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(selected_model + " Confusion Matrix")

            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

        except FileNotFoundError:
            st.error(
                "Model file not found. Please check whether all model files are available inside the model folder."
            )

        except Exception as e:
            st.error("Error while evaluating the model.")
            st.write(e)

    else:
        st.info("Please upload test_data.csv to evaluate the model.")


# ------------------------------------------------------------
# Section 3: Compare All Models
# ------------------------------------------------------------

elif section == "Compare All Models":

    st.header("Compare All Models")

    uploaded_file = st.file_uploader(
        "Upload test_data.csv",
        type=["csv"]
    )

    if uploaded_file is not None:

        test_df = pd.read_csv(uploaded_file)

        X_test, y_test = prepare_uploaded_data(test_df)

        comparison_results = []

        for model_name in model_files.keys():

            try:
                model = load_model(model_name)

                metrics, y_pred = evaluate_model(
                    model,
                    X_test,
                    y_test,
                    model_name
                )

                comparison_results.append({
                    "Model": model_name,
                    "Accuracy": metrics["Accuracy"],
                    "AUC Score": metrics["AUC Score"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1 Score": metrics["F1 Score"],
                    "MCC Score": metrics["MCC Score"]
                })

            except Exception as e:
                st.warning(f"Could not evaluate {model_name}: {e}")

        results_df = pd.DataFrame(comparison_results)

        st.subheader("Model Comparison Table")

        if not results_df.empty:

            st.dataframe(results_df.round(4))

            best_model = results_df.sort_values(
                by=["Accuracy", "AUC Score", "MCC Score"],
                ascending=False
            ).iloc[0]["Model"]

            st.success(
                "Best model based on Accuracy, AUC Score and MCC Score: "
                + best_model
            )

            # Accuracy Chart
            st.subheader("Accuracy Comparison")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(results_df["Model"], results_df["Accuracy"])
            ax.set_xlabel("Model")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy Comparison of Classification Models")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)

            # AUC Chart
            st.subheader("AUC Score Comparison")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(results_df["Model"], results_df["AUC Score"])
            ax.set_xlabel("Model")
            ax.set_ylabel("AUC Score")
            ax.set_title("AUC Score Comparison of Classification Models")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)

        else:
            st.error("No model could be evaluated. Please check saved model files.")

    else:
        st.info("Please upload test_data.csv to compare all models.")